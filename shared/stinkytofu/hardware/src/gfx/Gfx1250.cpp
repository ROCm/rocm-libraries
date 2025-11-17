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
* THE SOFTWARE IS PROVIDED "AS IS") WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */
#include <string>
#include <utility>

#include "gfx/CommonInstsDSL.hpp"
#include "gfx/GpuArchManager.hpp"
#include "gfx/InstDefDSL.hpp"

namespace stinkytofu
{
    uint16_t computeCdna5WmmaLatency(const WMMA& a)
    {
        // TODO: Implement detail latency calculation
        return 16;
    }

    // ---- gfx1250-only definitions ----
    void defineGfx1250Insts(GpuArch& registry)
    {
        defineGfx950Insts(registry);

        // gfx1250 removes ds_read/ds_write variant instructions.
        std::vector<std::string> removedInsts;
        for(const auto& inst : registry.getInstructions())
        {
            if(inst->is(IF_DSRead) || inst->is(IF_DSStore))
            {
                removedInsts.push_back(inst->name);
            }
        }

        for(const auto& name : removedInsts)
        {
            registry.erase(name);
        }

        // ============================================
        // TDM
        // ============================================
        DEF_T(TensorLoadToLds, "tensor_load_to_lds");

        DEF_T(WaitTensorCntInst, "s_wait_tensorcnt");

        // ============================================
        // DS: LDS access
        // ============================================
        for(std::string ty : {"b32", "b64", "b96", "b128", "u8", "i8", "u16", "i16"})
            DEF_T(DSRead, "ds_load_" + ty);

        for(std::string ty : {"b8", "b16", "b32", "b64", "b96", "b128"})
            DEF_T(DSWrite, "ds_store_" + ty);

        // ============================================
        // WMMA / SWMMA (matrix instructions)
        // ============================================
        const MatInstDesc wmma1250[] = {
            // V_WMMA_F32_16X16X32_BF16
            {16, 16, 32, 1, "f32", "bf16"},
        };

        for(auto s : wmma1250)
            GEN_WMMA(registry, s, false);
    }
} // namespace stinkytofu
