/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/config.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/batched_transpose_sol.hpp>
#include <miopen/buffer_info.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/miopen_internal.h>
#include <miopen/hip_build_utils.hpp>

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_CKTILE_COMPOSABLEKERNEL
// Include CK tile headers for convolution operations
#include <ck_tile/ops/grouped_convolution.hpp>
#include <ck_tile/ops/elementwise.hpp> // For PassThrough
#include <ck_tile/host/stream_config.hpp> // For stream_config

namespace miopen {
namespace solver {

// CK tile utility functions
namespace ck_tile_utility {
    static inline bool is_ck_tile_supported_hardware(const Handle& handle)
    {
        // CK tile supported hardware list
        return (StartsWith(handle.GetDeviceName(), "gfx1100") ||
               StartsWith(handle.GetDeviceName(), "gfx1101") ||
               StartsWith(handle.GetDeviceName(), "gfx1102") ||
               StartsWith(handle.GetDeviceName(), "gfx1200") ||
               StartsWith(handle.GetDeviceName(), "gfx1201"));
    }
    
    static inline bool is_ck_tile_whitelist(const std::string& device_name)
    {
        return (StartsWith(device_name, "gfx11") ||
                StartsWith(device_name, "gfx12"));
    }
    
    static inline bool is_ck_tile_whitelist(const Handle& handle)
    {
        return is_ck_tile_whitelist(handle.GetDeviceName());
    }
}//namespace ck_tile_utility

namespace conv_ck_tile { 

// Solvers using Channel-Last (NDHWC) data may need to transpose to these layouts.
using InLayout    = ck_tile::tensor_layout::convolution::NDHWGC;
using WeiLayout   = ck_tile::tensor_layout::convolution::GKZYXC;
using OutLayout   = ck_tile::tensor_layout::convolution::NDHWGK;
using PassThrough = ck_tile::element_wise::PassThrough; 

using StreamConfig = ck_tile::stream_config;



} // namespace conv_ck_tile
} // namespace solver
} // namespace miopen

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_CKTILE_COMPOSABLEKERNEL