/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

#pragma once
#include "InstDefDSL.hpp"

namespace stinkytofu
{
    // Instruction subclasses for different types and different architectures.
    //
    // Note: The subclass can define its own data and methods as needed for
    //       table generator.
    struct MUBUFLoad : GfxInstDef
    {
        MUBUFLoad()
        {
            hwInstDesc.flags.set(IF_MUBUFLoad);
        }
    };

    struct MUBUFStore : GfxInstDef
    {
        MUBUFStore()
        {
            hwInstDesc.flags.set(IF_MUBUFStore);
        }
    };

    struct MUBUFAtomic : GfxInstDef
    {
        MUBUFAtomic()
        {
            hwInstDesc.flags.set(IF_MUBUFAtomic);
        }
    };

    struct FLATLoad : GfxInstDef
    {
        FLATLoad()
        {
            hwInstDesc.flags.set(IF_FLATLoad);
        }
    };

    struct FLATStore : GfxInstDef
    {
        FLATStore()
        {
            hwInstDesc.flags.set(IF_FLATStore);
        }
    };

    struct FLATAtomic : GfxInstDef
    {
        FLATAtomic()
        {
            hwInstDesc.flags.set(IF_FLATAtomic);
        }
    };

    struct GLOBALLoad : GfxInstDef
    {
        GLOBALLoad()
        {
            hwInstDesc.flags.set(IF_GLOBALLoad);
        }
    };

    struct GLOBALStore : GfxInstDef
    {
        GLOBALStore()
        {
            hwInstDesc.flags.set(IF_GLOBALStore);
        }
    };

    struct TensorLoadToLds : GfxInstDef
    {
        TensorLoadToLds()
        {
            hwInstDesc.flags.set(IF_TENSORLoadToLds);
        }
    };

    struct SMemLoad : GfxInstDef
    {
        SMemLoad()
        {
            hwInstDesc.flags.set(IF_SMemLoad);
        }
    };

    struct SMemStore : GfxInstDef
    {
        SMemStore()
        {
            hwInstDesc.flags.set(IF_SMemStore);
        }
    };

    struct SMemAtomic : GfxInstDef
    {
        SMemAtomic()
        {
            hwInstDesc.flags.set(IF_SMemAtomic);
        }
    };

    struct DSRead : GfxInstDef
    {
        DSRead()
        {
            hwInstDesc.flags.set(IF_DSRead);
        }
    };

    struct DSWrite : GfxInstDef
    {
        DSWrite()
        {
            hwInstDesc.flags.set(IF_DSStore);
        }
    };

    struct DSAtomic : GfxInstDef
    {
        DSAtomic()
        {
            hwInstDesc.flags.set(IF_DSAtomic);
        }
    };

    struct BarrierInst : GfxInstDef
    {
        BarrierInst()
        {
            hwInstDesc.flags.set(IF_Barrier);
        }
    };

    struct BranchInst : GfxInstDef
    {
        BranchInst()
        {
            hwInstDesc.flags.set(IF_Branch);
        }
    };

    struct ConditionalBranchInst : BranchInst
    {
        ConditionalBranchInst()
        {
            hwInstDesc.flags.set(IF_ConditionalBranch);
        }
    };

    struct WaitCntInst : GfxInstDef
    {
        WaitCntInst()
        {
            hwInstDesc.flags.set(IF_WaitCnt);
        }
    };

    struct WaitTensorCntInst : WaitCntInst
    {
        WaitTensorCntInst()
        {
            hwInstDesc.flags.set(IF_WaitTensorCnt);
        }
    };

    struct HasSideEffectInst : GfxInstDef
    {
        HasSideEffectInst()
        {
            hwInstDesc.flags.set(IF_HasSideEffect);
        }
    };

    // Simple wrappers for DEF_T(class, "mnemonic") from generated .inc (default ctor only)
    struct MFMAInst : GfxInstDef
    {
        MFMAInst()
        {
            hwInstDesc.flags.set(IF_MFMA);
        }
    };
    struct SMFMAInst : GfxInstDef
    {
        SMFMAInst()
        {
            hwInstDesc.flags.set(IF_SMFMA);
        }
    };
    struct WMMAInst : GfxInstDef
    {
        WMMAInst()
        {
            hwInstDesc.flags.set(IF_WMMA);
        }
    };
    struct SWMMAInst : GfxInstDef
    {
        SWMMAInst()
        {
            hwInstDesc.flags.set(IF_SWMMA);
        }
    };
    struct MXWMMAInst : GfxInstDef
    {
        MXWMMAInst()
        {
            hwInstDesc.flags.set(IF_MXWMMA);
        }
    };

    struct MFMA : GfxInstDef
    {
        int M = 0, N = 0, K = 0, B = 0;

        std::string outTy, inTy;

        bool sparse = false;

        MFMA(int                m,
             int                n,
             int                k,
             int                b,
             const std::string& outT,
             const std::string& inT,
             bool               sparse)
            : M(m)
            , N(n)
            , K(k)
            , B(b)
            , outTy(outT)
            , inTy(inT)
            , sparse(sparse)
        {
            if(sparse)
                hwInstDesc.flags.set(IF_SMFMA);
            else
                hwInstDesc.flags.set(IF_MFMA);
        }
    };

    struct WMMA : GfxInstDef
    {
        int M = 0, N = 0, K = 0, B = 0;

        std::string outTy, inTy;

        bool sparse = false;

        WMMA(int                m,
             int                n,
             int                k,
             int                b,
             const std::string& outT,
             const std::string& inT,
             bool               sparse)
            : M(m)
            , N(n)
            , K(k)
            , B(b)
            , outTy(outT)
            , inTy(inT)
            , sparse(sparse)
        {
            if(sparse)
                hwInstDesc.flags.set(IF_SWMMA);
            else
                hwInstDesc.flags.set(IF_WMMA);
        }
    };

    struct MXWMMA : GfxInstDef
    {
        int M = 0, N = 0, K = 0, B = 0;
        int block = 32; // block size: 16 or 32

        std::string outTy, inTy;

        MXWMMA(int                m,
               int                n,
               int                k,
               int                b,
               const std::string& outT,
               const std::string& inT,
               int                blk = 32)
            : M(m)
            , N(n)
            , K(k)
            , B(b)
            , outTy(outT)
            , inTy(inT)
            , block(blk)
        {
            hwInstDesc.flags.set(IF_MXWMMA);
            hwInstDesc.flags.set(IF_WMMA); // Also set WMMA flag for compatibility
        }
    };

    // The following instruction types are placeholders for future use.
    struct SALU : GfxInstDef
    {
        SALU()
        {
            hwInstDesc.flags.set(IF_SALU);
        }
    };

    struct VALU : GfxInstDef
    {
        VALU()
        {
            hwInstDesc.flags.set(IF_VALU);
        }
    };

    // Transcendental VALU instructions (v_s_*, v_exp_*, v_log_*, v_rcp_*, v_rsq_*, v_sqrt_*)
    struct Transcendental : VALU
    {
        Transcendental()
        {
            hwInstDesc.flags.set(IF_Transcendental);
        }
    };

    // Commutative VALU instructions (add, mul, min, max, and, or, xor, etc.)
    struct CommutativeVALU : VALU
    {
        CommutativeVALU()
        {
            hwInstDesc.flags.set(IF_Commutative);
        }
    };

    struct SCmp : GfxInstDef
    {
    };

    struct VCmp : GfxInstDef
    {
        VCmp()
        {
            hwInstDesc.flags.set(IF_VCmp);
        }
    };

    struct VCmpX : VCmp
    {
        VCmpX()
        {
            hwInstDesc.flags.set(IF_VCmpX);
        }
    };

    struct VCvt : GfxInstDef
    {
    };

    struct VTrans : VALU
    {
        VTrans()
        {
            hwInstDesc.flags.set(IF_Transcendental);
        }
    };

    // fp8/bf8 scale-variants (pk/sr)
    struct VCvtScale : VCvt
    {
    };

    // vector dot ops
    struct VDot : VALU
    {
    };

    // MFMA/SMFMAC generation helper
    // Generate MFMA or SMFMAC instruction with proper file/line tracking
    struct MatInstDesc
    {
        int M, N, K, B;

        const char* outTy;
        const char* inTy;
    };

    inline MFMA* genMfmaImpl(
        GpuArch& registry, const MatInstDesc& s, bool sparse, const char* file, size_t line)
    {
        std::string mfmaName;
        if(!sparse)
            mfmaName = "v_mfma_";
        else
            mfmaName = "v_smfmac_";

        mfmaName = mfmaName + s.outTy + "_" + std::to_string(s.M) + "x" + std::to_string(s.N) + "x"
                   + std::to_string(s.K);
        if(s.B > 1)
            mfmaName = mfmaName + "_" + std::to_string(s.B) + "b";
        mfmaName = mfmaName + "_" + s.inTy;

        MFMA* mfmaInst = defT<MFMA>(
            mfmaName, registry, file, line, s.M, s.N, s.K, s.B, s.outTy, s.inTy, sparse);
        // Costs set by applyInstructionCosts() from architecture-specific cost tables

        return mfmaInst;
    }

    inline WMMA* genWmmaImpl(
        GpuArch& registry, const MatInstDesc& s, bool sparse, const char* file, size_t line)
    {
        std::string wmmaName;
        if(!sparse)
            wmmaName = "v_wmma_";
        else
            wmmaName = "v_swmmac_";

        wmmaName = wmmaName + s.outTy + "_" + std::to_string(s.M) + "x" + std::to_string(s.N) + "x"
                   + std::to_string(s.K) + +"_" + s.inTy;

        WMMA* wmmaInst = defT<WMMA>(
            wmmaName, registry, file, line, s.M, s.N, s.K, s.B, s.outTy, s.inTy, sparse);

        // Costs set by applyInstructionCosts() from architecture-specific cost tables

        return wmmaInst;
    }

    inline MXWMMA* genMxWmmaImpl(
        GpuArch& registry, const MatInstDesc& s, int block, const char* file, size_t line)
    {
        std::string mxwmmaName = "v_wmma_scale";
        if(block == 16)
            mxwmmaName += "16";

        mxwmmaName = mxwmmaName + "_" + s.outTy + "_" + std::to_string(s.M) + "x"
                     + std::to_string(s.N) + "x" + std::to_string(s.K) + "_" + s.inTy;

        MXWMMA* mxwmmaInst = defT<MXWMMA>(
            mxwmmaName, registry, file, line, s.M, s.N, s.K, s.B, s.outTy, s.inTy, block);

        // Costs set by applyInstructionCosts() from architecture-specific cost tables

        return mxwmmaInst;
    }

// Macro wrapper to capture __FILE__ and __LINE__ from call site
#define GEN_MFMA(registry, desc, sparse) genMfmaImpl(registry, desc, sparse, __FILE__, __LINE__)
#define GEN_WMMA(registry, desc, sparse) genWmmaImpl(registry, desc, sparse, __FILE__, __LINE__)
#define GEN_MXWMMA(registry, desc, block) genMxWmmaImpl(registry, desc, block, __FILE__, __LINE__)

} // namespace stinkytofu
