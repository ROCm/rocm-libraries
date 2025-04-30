// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef RTC_STOCKHAM
#define RTC_STOCKHAM

#include <vector>

#include "../device/generator/stockham_gen.h"
#include "compute_scheme.h"
#include "load_store_ops.h"
#include "rocfft/rocfft.h"
#include "rtc_kernel.h"

#include "../device/kernels/common.h"

// generate name for RTC stockham kernel
std::string stockham_rtc_kernel_name(const StockhamGeneratorSpecs& specs,
                                     const StockhamGeneratorSpecs& specs2d,
                                     ComputeScheme                 scheme,
                                     int                           direction,
                                     rocfft_precision              precision,
                                     rocfft_result_placement       placement,
                                     rocfft_array_type             inArrayType,
                                     rocfft_array_type             outArrayType,
                                     bool                          unitstride,
                                     size_t                        largeTwdBase,
                                     size_t                        largeTwdSteps,
                                     bool                          largeTwdBatchIsTransformCount,
                                     EmbeddedType                  ebtype,
                                     DirectRegType                 dir2regMode,
                                     IntrinsicAccessType           intrinsicMode,
                                     SBRC_TRANSPOSE_TYPE           transpose_type,
                                     CallbackType                  cbtype,
                                     BluesteinFuseType             fuseBlue,
                                     const LoadOps&                loadOps,
                                     const StoreOps&               storeOps);

// generate source for RTC stockham kernel.  transforms_per_block may
// be nullptr, but if non-null, stockham_rtc stores the number of
// transforms each threadblock will do
std::string stockham_rtc(const StockhamGeneratorSpecs& specs,
                         const StockhamGeneratorSpecs& specs2d,
                         unsigned int*                 transforms_per_block,
                         const std::string&            kernel_name,
                         ComputeScheme                 scheme,
                         int                           direction,
                         rocfft_precision              precision,
                         rocfft_result_placement       placement,
                         rocfft_array_type             inArrayType,
                         rocfft_array_type             outArrayType,
                         bool                          unit_stride,
                         size_t                        largeTwdBase,
                         size_t                        largeTwdSteps,
                         bool                          largeTwdBatchIsTransformCount,
                         EmbeddedType                  ebtype,
                         DirectRegType                 dir2regMode,
                         IntrinsicAccessType           intrinsicMode,
                         SBRC_TRANSPOSE_TYPE           transpose_type,
                         CallbackType                  cbtype,
                         const BluesteinFuseType&      fuseBlue,
                         const LoadOps&                loadOps,
                         const StoreOps&               storeOps);

#endif
