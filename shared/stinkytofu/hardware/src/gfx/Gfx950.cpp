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
    // fp8/bf8 scale-variants (pk/sr)
    struct VCvtScale : VCvt
    {
    };

    // vector dot ops
    struct VDot : VALU
    {
    };

    // ---- gfx950-only definitions (12 new mnemonics) ----
    void defineGfx950Insts(GpuArch& registry)
    {
        defineGfx942Insts(registry);

        // Set wavefront size for gfx950
        registry.setWaveFrontSize(64);

        // gfx950 removes xf32 variants of v_mfma.
        for(auto removed : {"v_mfma_f32_16x16x8_xf32", "v_mfma_f32_32x32x4_xf32"})
            registry.erase(removed);

        // 1) New plain/pack converts
        //    v_cvt_f32_bf16
        DEF_T(VCvt, "v_cvt_f32_bf16");

        //    v_cvt_pk_bf16_f32
        DEF_T(VCvt, "v_cvt_pk_bf16_f32");

        // 2) “scale” fp8/bf8 variants
        //    v_cvt_scalef32_f16_fp8
        DEF_T(VCvtScale, "v_cvt_scalef32_f16_fp8");

        //    pk_* pairs: (dst, src) ordered as they appear in mnemonics
        for(auto p : {
                std::pair{"bf8", "f16"},
                std::pair{"f16", "bf8"},
                std::pair{"f16", "fp8"},
                std::pair{"fp8", "f16"},
            })
        {
            std::string name = "v_cvt_scalef32_pk_";
            name += p.first;
            name += "_";
            name += p.second;
            DEF_T(VCvtScale, name.c_str());
        }

        //    sr_* pairs
        for(auto p : {
                std::pair{"bf8", "f16"},
                std::pair{"fp8", "f16"},
            })
        {
            std::string name = "v_cvt_scalef32_sr_";
            name += p.first;
            name += "_";
            name += p.second;
            DEF_T(VCvtScale, name.c_str());
        }

        // 3) New bf16 dot ops
        //    v_dot2_f32_bf16, v_dot2c_f32_bf16
        for(auto base : {"v_dot2_f32_", "v_dot2c_f32_"})
        {
            std::string name = std::string(base) + "bf16";
            DEF_T(VDot, name.c_str());
        }

        // 4) PRNG
        DEF_T(VALU, "v_prng_b32");

        // ds read b64 tr b16
        DEF_T(DSRead, "ds_read_b64_tr_b4");
        DEF_T(DSRead, "ds_read_b96_tr_b6");
        DEF_T(DSRead, "ds_read_b64_tr_b8");
        DEF_T(DSRead, "ds_read_b64_tr_b16");

        // ============================================
        // MFMA / SMFMAC (matrix instructions)
        // ============================================
        const MatInstDesc mfma950[] = {
            // V_MFMA_F32_16x16x128_F8F6F4
            {16, 16, 128, 1, "f32", "f8f6f4"},
            // V_MFMA_F32_32x32x64_F8F6F4
            {32, 32, 64, 1, "f32", "f8f6f4"},
        };

        for(auto s : mfma950)
            GEN_MFMA(registry, s, false);
    }
} // namespace stinkytofu
