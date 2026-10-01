/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#ifndef ROCWMMA_GEMM_KERNEL_BASE_INSTANTIATIONS_HPP
#define ROCWMMA_GEMM_KERNEL_BASE_INSTANTIATIONS_HPP

#include "gemm_kernel_base_impl.hpp"

// Architecture-gated input shards also use architecture-independent output
// kernels. For example, an FP8 shard instantiates float output helpers on the
// host but not on gfx942. HIP can then associate their coalesced host stubs with
// a code object that does not contain the device kernels. Give those helpers a
// single owner in the unconditional float32 shard instead. If a gated shard
// gains another output type, declare its helpers here and define them in an
// unconditional shard for that type as well.
#define ROCWMMA_INSTANTIATE_GEMM_OUTPUT_KERNELS(Prefix, DataT, Layout)                     \
    Prefix template __global__ void fillKernel<DataT, Layout>(DataT*, uint32_t, uint32_t); \
    Prefix template __global__ void fillValKernel<DataT, Layout>(                          \
        DataT*, uint32_t, uint32_t, DataT);                                                \
    Prefix template __global__ void compareEqualKernel<DataT, DataT, Layout, row_major>(   \
        DataT*, DataT*, float64_t*, uint32_t, uint32_t, uint32_t, uint32_t);               \
    Prefix template __global__ void compareEqualKernel<DataT, DataT, Layout, col_major>(   \
        DataT*, DataT*, float64_t*, uint32_t, uint32_t, uint32_t, uint32_t);

namespace rocwmma
{
    ROCWMMA_INSTANTIATE_GEMM_OUTPUT_KERNELS(extern, float32_t, row_major)
    ROCWMMA_INSTANTIATE_GEMM_OUTPUT_KERNELS(extern, float32_t, col_major)
} // namespace rocwmma

#define ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(InputT, OutputT, ComputeT) \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   4u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   256u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   2u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   8u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   4u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   256u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   2u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   8u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   4u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   256u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   2u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   8u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   4u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   256u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   2u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   8u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   4u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   256u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   2u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   8u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   4,                                   \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   256u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   2u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   8u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   row_major,                           \
                                   row_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   4u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   256u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   2u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   8u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   row_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   4u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<16u,                                 \
                                   16u,                                 \
                                   256u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   2u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   8u,                                  \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   16u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   32u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   64u,                                 \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;                          \
    template struct GemmKernelBase<32u,                                 \
                                   32u,                                 \
                                   128u,                                \
                                   InputT,                              \
                                   OutputT,                             \
                                   ComputeT,                            \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major,                           \
                                   col_major>;


#endif // ROCWMMA_GEMM_KERNEL_BASE_INSTANTIATIONS_HPP
