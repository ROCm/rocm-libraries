/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "gfx/LatencyHelper.hpp"
#include "gfx/CommonInstsDSL.hpp"

#include <cassert>
#include <string>

namespace stinkytofu
{
    uint16_t computeCdna3MfmaLatency(const MFMA& a)
    {
        auto isFp8Family = [](const std::string& ty) -> bool {
            // The fp8 type is "BF8_BF8" .. etc, match the "xx8_" prefix
            if(ty.size() < 4)
            {
                assert(ty != "bf8" && ty != "fp8" && "Invalid fp8 family type!");
                return false;
            }

            std::string prefix = ty.substr(0, 4);
            return (prefix == "bf8_" || prefix == "fp8_");
        };

        auto speedup = [&](const std::string& ty) -> uint16_t {
            if(ty == "f32" || ty == "f64")
                return 1;
            if(ty == "xf32")
                return 4;
            if(ty == "f16" || ty == "bf16")
                return 8;
            if(ty == "i8" || isFp8Family(ty))
                return 16;
            if(ty == "f8f6f4")
                return 16;

            assert(false && "Unknown MFMA input type for latency computation!");
            return 1;
        };

        uint16_t speed    = speedup(a.inTy);
        uint16_t sparsity = a.sparse ? 2 : 1;
        uint16_t slowdown = 1;

        if(a.B > 1)
        {
            if(a.inTy == "f64" || a.inTy == "f16" || a.inTy == "bf16")
                slowdown = 2;
            if(a.inTy == "i8")
                slowdown = 4;
        }

        uint16_t latency = (a.M * a.N * a.K * a.B * slowdown) / (speed * sparsity) / 32;
        return latency;
    }

    uint16_t computeCdna5WmmaLatency(const WMMA& a)
    {
        // TODO: Implement detail latency calculation
        return 16;
    }

} // namespace stinkytofu
