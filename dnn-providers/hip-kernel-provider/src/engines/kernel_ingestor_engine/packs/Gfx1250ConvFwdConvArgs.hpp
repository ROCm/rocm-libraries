// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdTypes.hpp"

/**
 * @file Gfx1250ConvFwdConvArgs.hpp
 * @brief The forward conv kernel ABIs and their values, a port of rocKE's ConvArgs.
 *
 * Every argument name is self-describing (kernels/common/conv_args.py): a stride is
 * `p_<tensor>_stride_<axis>`, a magic pair `p_magic_<m|k>_<divisor>_<mult|shift>`, and every
 * other scalar means one thing in every ABI. The two forward ABIs -- implicit GEMM
 * (kernels/common/conv_abi.conv_args_signature, 2-D) and direct conv
 * (conv_direct_args_signature) -- are the name lists below, in ABI order.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

/// The extents the arguments are computed from (rocKE's ConvGeometry, 2-D).
struct ConvGeometry
{
    int64_t n, hi, wi, c, k, y, x, sH, sW, pH, pW, dH, dW, groups;

    static ConvGeometry of(const Problem& p)
    {
        return {p.n,
                p.h,
                p.w,
                p.c,
                p.k,
                p.y,
                p.x,
                p.strideH,
                p.strideW,
                p.padH,
                p.padW,
                p.dilationH,
                p.dilationW,
                p.groups};
    }

    int64_t ho() const
    {
        return floorDiv(hi + 2 * pH - dH * (y - 1) - 1, sH) + 1;
    }
    int64_t wo() const
    {
        return floorDiv(wi + 2 * pW - dW * (x - 1) - 1, sW) + 1;
    }
    int64_t cpg() const
    {
        return c / groups;
    }
    int64_t kpg() const
    {
        return k / groups;
    }
    /// Forward GEMM extents: output positions (M) and filter-channels (K).
    int64_t gemmM() const
    {
        return n * ho() * wo();
    }
    int64_t gemmK() const
    {
        return y * x * cpg();
    }
};

inline const std::vector<std::string>& implicitFwdArgNames()
{
    static const std::vector<std::string> s_names = {"A",
                                                     "B",
                                                     "D",
                                                     "A_bytes",
                                                     "B_bytes",
                                                     "D_bytes",
                                                     "p_N",
                                                     "p_Hi",
                                                     "p_Wi",
                                                     "p_C",
                                                     "p_K",
                                                     "p_Y",
                                                     "p_X",
                                                     "p_sH",
                                                     "p_sW",
                                                     "p_pH",
                                                     "p_pW",
                                                     "p_dH",
                                                     "p_dW",
                                                     "p_groups",
                                                     "p_Ho",
                                                     "p_Wo",
                                                     "p_cpg",
                                                     "p_kpg",
                                                     "p_K_gemm",
                                                     "p_M",
                                                     "p_A_stride_n",
                                                     "p_A_stride_hi",
                                                     "p_A_stride_wi",
                                                     "p_B_stride_k",
                                                     "p_B_stride_y",
                                                     "p_B_stride_x",
                                                     "p_D_stride_n",
                                                     "p_D_stride_ho",
                                                     "p_D_stride_wo",
                                                     "p_magic_m_Ho_mult",
                                                     "p_magic_m_Ho_shift",
                                                     "p_magic_m_Wo_mult",
                                                     "p_magic_m_Wo_shift",
                                                     "p_magic_k_X_mult",
                                                     "p_magic_k_X_shift",
                                                     "p_magic_k_cpg_mult",
                                                     "p_magic_k_cpg_shift",
                                                     "p_num_pid_m",
                                                     "p_num_pid_n"};
    return s_names;
}

inline const std::vector<std::string>& directFwdArgNames()
{
    static const std::vector<std::string> s_names = {"A",
                                                     "B",
                                                     "D",
                                                     "A_bytes",
                                                     "B_bytes",
                                                     "D_bytes",
                                                     "p_N",
                                                     "p_Hi",
                                                     "p_Wi",
                                                     "p_Ho",
                                                     "p_Wo",
                                                     "p_groups",
                                                     "p_total_c",
                                                     "p_total_k",
                                                     "p_A_stride_n",
                                                     "p_A_stride_hi",
                                                     "p_A_stride_wi",
                                                     "p_D_stride_n",
                                                     "p_D_stride_ho",
                                                     "p_D_stride_wo"};
    return s_names;
}

/// The value of argument @p name for @p g; @p tileM / @p tileN are the implicit-GEMM
/// instance's tile (0 for direct conv, which has no GEMM tile counts).
inline Arg convArg(const std::string& name, const ConvGeometry& g, int64_t tileM, int64_t tileN)
{
    struct Pointer
    {
        const char* name;
        const char* buffer;
    };
    const std::array<Pointer, 3> pointers = {{{"A", "x"}, {"B", "w"}, {"D", "y"}}};
    for(const auto& p : pointers)
    {
        if(name == p.name)
        {
            return Arg::pointer(name, p.buffer);
        }
        if(name == std::string(p.name) + "_bytes")
        {
            return Arg::bytes(name, p.buffer);
        }
    }
    const auto ho = g.ho();
    const auto wo = g.wo();
    struct Scalar
    {
        const char* name;
        int64_t value;
    };
    const std::initializer_list<Scalar> scalars = {
        {"p_N", g.n},
        {"p_Hi", g.hi},
        {"p_Wi", g.wi},
        {"p_C", g.c},
        {"p_K", g.k},
        {"p_Y", g.y},
        {"p_X", g.x},
        {"p_sH", g.sH},
        {"p_sW", g.sW},
        {"p_pH", g.pH},
        {"p_pW", g.pW},
        {"p_dH", g.dH},
        {"p_dW", g.dW},
        {"p_groups", g.groups},
        {"p_Ho", ho},
        {"p_Wo", wo},
        {"p_cpg", g.cpg()},
        {"p_kpg", g.kpg()},
        {"p_total_c", g.c},
        {"p_total_k", g.k},
        {"p_M", g.gemmM()},
        {"p_K_gemm", g.gemmK()},
        // Activation (NHWC), output (NHWK) and filter (KYXC, per-group channels) strides.
        {"p_A_stride_n", g.hi * g.wi * g.c},
        {"p_A_stride_hi", g.wi * g.c},
        {"p_A_stride_wi", g.c},
        {"p_D_stride_n", ho * wo * g.k},
        {"p_D_stride_ho", wo * g.k},
        {"p_D_stride_wo", g.k},
        {"p_B_stride_k", g.y * g.x * g.cpg()},
        {"p_B_stride_y", g.x * g.cpg()},
        {"p_B_stride_x", g.cpg()},
    };
    for(const auto& s : scalars)
    {
        if(name == s.name)
        {
            return Arg::i32(name, s.value);
        }
    }
    struct Magic
    {
        const char* prefix;
        int64_t divisor;
    };
    const std::initializer_list<Magic> magics = {{"p_magic_m_Ho_", ho},
                                                 {"p_magic_m_Wo_", wo},
                                                 {"p_magic_k_X_", g.x},
                                                 {"p_magic_k_cpg_", g.cpg()}};
    for(const auto& m : magics)
    {
        const auto magic = magicNumbers(m.divisor);
        if(name == std::string(m.prefix) + "mult")
        {
            return Arg::i32(name, magic.multiplier);
        }
        if(name == std::string(m.prefix) + "shift")
        {
            return Arg::i32(name, magic.shift);
        }
    }
    if(tileM > 0 && name == "p_num_pid_m")
    {
        return Arg::i32(name, ceilDiv(g.gemmM(), tileM));
    }
    if(tileN > 0 && name == "p_num_pid_n")
    {
        return Arg::i32(name, ceilDiv(g.kpg(), tileN));
    }
    throw std::logic_error("no host-side rule for conv kernarg '" + name + "'");
}

inline std::vector<Arg> implicitFwdArgs(const ConvGeometry& g, int64_t tileM, int64_t tileN)
{
    std::vector<Arg> out;
    for(const auto& name : implicitFwdArgNames())
    {
        out.push_back(convArg(name, g, tileM, tileN));
    }
    return out;
}

inline std::vector<Arg> directFwdArgs(const ConvGeometry& g)
{
    std::vector<Arg> out;
    for(const auto& name : directFwdArgNames())
    {
        out.push_back(convArg(name, g, 0, 0));
    }
    return out;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
