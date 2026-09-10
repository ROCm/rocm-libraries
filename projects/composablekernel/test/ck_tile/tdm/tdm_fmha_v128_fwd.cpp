// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <string_view>
#include "tdm_fmha_v128_test_common.hpp"

#ifdef CK_TILE_FMHA_TDM_V128_TEST_ORACLE
#include "tdm_fmha_v128_runtime_reference.hpp"
#endif

#ifndef CK_TILE_FMHA_TDM_V128_TEST_PROFILE
#define CK_TILE_FMHA_TDM_V128_TEST_PROFILE 0
#endif

namespace {

constexpr int kProfile = CK_TILE_FMHA_TDM_V128_TEST_PROFILE;
static_assert(kProfile >= 0 && kProfile <= 6);

template <bool Group, bool Mask, bool Lse>
float Launch(const fmha_fwd_args& args, const ck_tile::stream_config& stream)
{
    using Problem  = typename tdm_v128_test::Model<128, Group, Mask, Lse>::Problem;
    using Policy   = tdm_v128_test::Policy<Problem, kProfile>;
    using Pipeline = ck_tile::BlockFmhaPipelineQRKSVSTdmV128<Problem, Policy>;
    using Epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<float, ck_tile::bf16_t, true, true>>;
    using Kernel = ck_tile::FmhaFwdKernel<Pipeline, Epilogue>;
    if(stream.log_level_ > 0)
        std::cout << ", fmha_fwd_d128_bf16_" << (Group ? "group" : "batch")
                  << "_b128x128x32x128x32x128_qr_tdm_v128_" << (Mask ? "mask" : "nmask")
                  << (Lse ? "_lse" : "_nlse") << "_test_profile" << kProfile << std::flush;
    auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<Kernel>(args);
    return ck_tile::launch_kernel(stream,
                                  ck_tile::make_kernel<Kernel::kBlockPerCu, ck_tile::gfx125_t>(
                                      Kernel{}, grids, Kernel::BlockSize(), 0, kargs));
}

template <bool Group, bool Mask>
float SelectLse(bool lse, const fmha_fwd_args& args, const ck_tile::stream_config& stream)
{
    return lse ? Launch<Group, Mask, true>(args, stream) : Launch<Group, Mask, false>(args, stream);
}

template <bool Group>
float SelectMask(const fmha_fwd_traits& traits,
                 const fmha_fwd_args& args,
                 const ck_tile::stream_config& stream)
{
    return traits.mask_type == mask_enum::no_mask
               ? SelectLse<Group, false>(traits.has_lse, args, stream)
               : SelectLse<Group, true>(traits.has_lse, args, stream);
}

} // namespace

// This symbol belongs only to the standalone test executable. No generated fallback is linked.
float fmha_fwd(fmha_fwd_traits traits, fmha_fwd_args args, const ck_tile::stream_config& stream)
{
    int device = 0;
    hipDeviceProp_t properties{};
    ck_tile::hip_check_error(hipGetDevice(&device));
    ck_tile::hip_check_error(hipGetDeviceProperties(&properties, device));
    if(!std::string_view{properties.gcnArchName}.starts_with("gfx1250"))
        std::exit(77);

    if(traits.hdim_q != 128 || traits.hdim_v != 128 || args.hdim_q != 128 || args.hdim_v != 128 ||
       traits.data_type != "bf16" || !traits.is_v_rowmajor || traits.has_logits_soft_cap ||
       traits.bias_type != bias_enum::no_bias || traits.has_dropout || traits.has_sink ||
       traits.qscale_type != quant_scale_enum::no_scale ||
       traits.mask_type == mask_enum::window_generic || args.nhead_k <= 0 ||
       args.nhead_q % args.nhead_k != 0)
        return -1;

    const float elapsed = traits.is_group_mode ? SelectMask<true>(traits, args, stream)
                                               : SelectMask<false>(traits, args, stream);
#ifdef CK_TILE_FMHA_TDM_V128_TEST_ORACLE
    if(elapsed >= 0)
    {
        const auto validation = tdm_v128_test::reference::ValidateFromDevice(traits, args);
        std::cout << '\n' << validation.report << std::endl;
        if(!validation.metrics.passed)
            throw std::runtime_error("independent FP32 attention oracle failed");
    }
#endif
    return elapsed;
}
