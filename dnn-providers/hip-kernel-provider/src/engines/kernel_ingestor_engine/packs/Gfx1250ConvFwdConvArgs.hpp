// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
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

/// The kernarg names of an ABI: the leading pointer/size block, then @p block. The blocks
/// are the shared conv_aot lists (ConvAotGeometryCommon.hpp), which the gfx950 port
/// matches the same way.
template <size_t N>
std::vector<std::string> fwdArgNames(const std::array<std::string_view, N>& block)
{
    std::vector<std::string> names(conv_aot::FWD_LEADING_KERNARG_NAMES.begin(),
                                   conv_aot::FWD_LEADING_KERNARG_NAMES.end());
    names.insert(names.end(), block.begin(), block.end());
    return names;
}

inline const std::vector<std::string>& implicitFwdArgNames()
{
    static const auto s_names = fwdArgNames(conv_aot::IMPLICIT_GEMM_KERNARG_NAMES);
    return s_names;
}

inline const std::vector<std::string>& directFwdArgNames()
{
    static const auto s_names = fwdArgNames(conv_aot::DIRECT_KERNARG_NAMES);
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
        const bool isMult = name == std::string(m.prefix) + "mult";
        if(!isMult && name != std::string(m.prefix) + "shift")
        {
            continue;
        }
        // Refused as a plan by ImplicitGemm::refuse before an argument is built.
        const auto magic = conv_aot::magicDivision(m.divisor);
        if(!magic)
        {
            throw std::invalid_argument("magic-division divisor " + std::to_string(m.divisor)
                                        + " is outside [1, 2^31)");
        }
        return Arg::i32(name, isMult ? magic->multiplier : magic->shift);
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
