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

#include "gemm_kernel_base_impl.hpp"

namespace rocwmma
{
    bool KernelI::sHeaderPrinted = false;

    // All supported instantiations
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(int8_t, int32_t, int32_t);

#if ROCWMMA_FP8
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(float8_t, float32_t, float32_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(bfloat8_t, float32_t, float32_t);
#endif

#if ROCWMMA_FP8_FNUZ
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(float8_fnuz_t, float32_t, float32_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(bfloat8_fnuz_t, float32_t, float32_t);
#endif

    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(bfloat16_t, float32_t, float32_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(float16_t, float32_t, float32_t);

#if !ROCWMMA_TESTS_NO_HALF
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(hfloat16_t, float32_t, float32_t);
#endif // !ROCWMMA_TESTS_NO_HALF

    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(float32_t, float32_t, float32_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(xfloat32_t, float32_t, float32_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(float64_t, float64_t, float64_t);

#if ROCWMMA_EXTENDED_TESTS
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(int8_t, int8_t, int32_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(bfloat16_t, bfloat16_t, bfloat16_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(bfloat16_t, bfloat16_t, float32_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(float16_t, float16_t, float16_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(float16_t, float16_t, float32_t);
#if !ROCWMMA_TESTS_NO_HALF
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(hfloat16_t, hfloat16_t, hfloat16_t);
    ROCWMMA_INSTANTIATE_GEMM_KERNEL_BASE(hfloat16_t, hfloat16_t, float32_t);
#endif // !ROCWMMA_TESTS_NO_HALF
#endif // ROCWMMA_EXTENDED_TESTS

} // namespace rocwmma
