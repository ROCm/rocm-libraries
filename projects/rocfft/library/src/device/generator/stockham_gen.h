// Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

// interface for generation of stockham kernels

#pragma once
#include "../../../../shared/arithmetic.h"
#include "../kernels/device_enum.h"
#include "rocfft/rocfft.h"
#include <ostream>
#include <string>
#include <vector>

struct StockhamGeneratorSpecs
{
    StockhamGeneratorSpecs(const std::vector<unsigned int>& factors,
                           const std::vector<unsigned int>& factors2d,
                           const std::vector<unsigned int>& precisions,
                           unsigned int                     workgroup_size,
                           const std::string&               scheme)
        : factors(factors)
        , factors2d(factors2d)
        , precisions(precisions)
        , length(product(factors.begin(), factors.end()))
        , length2d(product(factors2d.begin(), factors2d.end()))
        , workgroup_size(workgroup_size)
        , scheme(scheme)
    {
    }

    std::vector<unsigned int> factors;
    std::vector<unsigned int> factors2d;
    std::vector<unsigned int> precisions; // mapped from rocfft_precision
    unsigned int              length;
    unsigned int              length2d = 0;

    unsigned int workgroup_size;
    unsigned int threads_per_transform = 0;
    bool         half_lds              = false;
    bool         direct_to_from_reg    = false;
    // dimension of the kernel - 0 if the generated kernel accepts a
    // 'dim' argument at runtime; otherwise the dimension is
    // statically defined for the kernel
    unsigned int static_dim = 0;
    std::string  scheme;

    EmbeddedType ebtype = EmbeddedType::NONE;

    // this value indicating if the wgs, tpt are excatly what we want
    // (i.e. were already derived somewhere)
    // to tell StockhamKernel not to do its auto-derivation again.
    // Particularly useful when tuning or running a tuned kernel. (RTC-ing)
    // We don't want them to be overwritten by StockhamKernel.
    bool wgs_is_derived = false;

    // by default, aim for occupancy 2 assuming device has 64kiB LDS
    unsigned int lds_byte_limit = 32 * 1024;
    // by default, assume double-precision complex elements when
    // computing how much FFT data will fit into LDS
    unsigned int bytes_per_element = 16;
};

// generate default stockham variants for ahead-of-time compilation
void stockham_variants(const std::vector<std::string>& kernel_name,
                       const StockhamGeneratorSpecs&   specs,
                       const StockhamGeneratorSpecs&   specs2d,
                       std::ostream&                   output);

struct StockhamPartialPassParams
{
    StockhamPartialPassParams() = default;

    StockhamPartialPassParams(const std::vector<unsigned int>& parent_length,
                              const unsigned int               current_dim,
                              const unsigned int               off_dim,
                              const std::vector<unsigned int>& pp_factors_curr,
                              const std::vector<unsigned int>& pp_factors_other)
        : parent_length(parent_length)
        , current_dim(current_dim)
        , off_dim(off_dim)
        , pp_factors_curr(pp_factors_curr)
        , pp_factors_other(pp_factors_other)
    {
    }

    std::vector<unsigned int> parent_length;
    unsigned int              current_dim = 0;
    unsigned int              off_dim     = 0;
    std::vector<unsigned int> pp_factors_curr;
    std::vector<unsigned int> pp_factors_other;
};

void stockham_partial_pass_variants(const std::string&               kernel_name,
                                    const StockhamGeneratorSpecs&    specs1,
                                    const StockhamGeneratorSpecs&    specs2,
                                    const StockhamPartialPassParams& params_1,
                                    const StockhamPartialPassParams& params_2,
                                    std::ostream&                    output);
