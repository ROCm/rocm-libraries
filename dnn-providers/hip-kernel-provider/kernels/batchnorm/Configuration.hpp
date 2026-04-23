// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "DefaultConfiguration.hpp"

#include "Bfloat16Dev.hpp"
#include "VectorTypes.hpp"
#include <type_traits>

// hip_kernel_provider root configs
namespace hip_kernel_provider
{

enum class type_strategy : int
{
    fp16,
    fp32,
    fpmix,
    bfpmix,
};

enum class neuron_op_type : int
{
    pasthru = 0, // x
    relu = 3, // max(0, x)
    clipped_relu = 7, // min(alpha, max(0, x))
    clamp = 10,
};

namespace detail
{

template <int LayoutNHWC,
          int SaveMeanVariance,
          int RunningResult,
          int UseFp16,
          int UseFp32,
          int UseFpmix,
          int UseBfpmix,
          int UseAMDGCN,
          int NrnOpId>
struct proto_config
{
    static_assert(LayoutNHWC == 0 || LayoutNHWC == 1,
                  "LayoutNHWC (HIP_PLUGIN_LAYOUT_NHWC) must be 0 or 1");
    static_assert(SaveMeanVariance == 0 || SaveMeanVariance == 1,
                  "SaveMeanVariance must be 0 or 1");
    static_assert(RunningResult == 0 || RunningResult == 1, "SaveMeanVariance must be 0 or 1");
    static_assert(UseFp16 == 0 || UseFp16 == 1, "UseFp16 must be 0 or 1");
    static_assert(UseFp32 == 0 || UseFp32 == 1, "UseFp32 must be 0 or 1");
    static_assert(UseFpmix == 0 || UseFpmix == 1, "UseFpmix must be 0 or 1");
    static_assert(UseBfpmix == 0 || UseBfpmix == 1, "UseBfpmix must be 0 or 1");
    static_assert((UseFp16 + UseFp32 + UseFpmix + UseBfpmix) == 1,
                  "only one of these configs can and must be chosen.");
    static_assert(UseAMDGCN == 0 || UseAMDGCN == 1, "UseAMDGCN must be 0 or 1");
    static_assert(NrnOpId >= 0 && NrnOpId <= 10,
                  "NrnOpId can only be interger between 0-10 (inclusive)");

    static constexpr bool layout_nhwc = static_cast<bool>(LayoutNHWC);
    static constexpr bool save_mean_variance = static_cast<bool>(SaveMeanVariance);
    static constexpr bool running_result = static_cast<bool>(RunningResult);
    static constexpr type_strategy input_type_strategy
        = UseFp16 ? type_strategy::fp16
                  : (UseFp32 ? type_strategy::fp32
                             : (UseFpmix ? type_strategy::fpmix : type_strategy::bfpmix));
    static constexpr bool use_amdgcn = UseAMDGCN;
    static constexpr auto neuron_op = static_cast<neuron_op_type>(NrnOpId);
};

} // namespace detail

using config = detail::proto_config<HIP_PLUGIN_LAYOUT_NHWC,
                                    HIP_PLUGIN_SAVE_MEAN_VARIANCE,
                                    HIP_PLUGIN_RUNNING_RESULT,
                                    HIP_PLUGIN_USE_FP16,
                                    HIP_PLUGIN_USE_FP32,
                                    HIP_PLUGIN_USE_FPMIX,
                                    HIP_PLUGIN_USE_BFPMIX,
                                    HIP_PLUGIN_USE_AMDGCN,
                                    HIP_PLUGIN_NRN_OP_ID>;

} // namespace hip_kernel_provider

// hip_kernel_provider batchnorm configs
namespace hip_kernel_provider
{

namespace batchnorm
{

enum class architecture : int
{
    unknown,
    gfx103x,
    gfx110x,
    gfx115x,
    gfx120x,
};

namespace detail
{

// TODO: why this is here, becasue before c++ 20, double is not supported to be template parameter
struct half_max
{
    static constexpr double value = HALF_MAX;
};

// TODO: why this is here, becasue before c++ 20, double is not supported to be template parameter
struct flt_max
{
    static constexpr double value = FLT_MAX;
};

template <int Grp0, int Grp1, int Grp2>
struct launch_dimension
{
    static_assert(Grp0 >= 0, "HIP_PLUGIN_BN_GRP0 should be always >= 0");
    static_assert(Grp1 >= 0, "HIP_PLUGIN_BN_GRP1 should be always >= 0");
    static_assert(Grp2 >= 0, "HIP_PLUGIN_BN_GRP2 should be always >= 0");
    static constexpr unsigned int grp0 = static_cast<unsigned int>(Grp0);
    static constexpr unsigned int grp1 = static_cast<unsigned int>(Grp1);
    static constexpr unsigned int grp2 = static_cast<unsigned int>(Grp2);
};

template <int Gfx103x, int Gfx110x, int Gfx120x, int Gfx115x>
struct architecture_switch
{
    static_assert(Gfx103x == 0 || Gfx103x == 1, "Gfx103x must be 0 or 1");
    static_assert(Gfx110x == 0 || Gfx110x == 1, "Gfx110x must be 0 or 1");
    static_assert(Gfx120x == 0 || Gfx120x == 1, "Gfx120x must be 0 or 1");
    static_assert(Gfx115x == 0 || Gfx115x == 1, "Gfx115x must be 0 or 1");
    static_assert(Gfx103x + Gfx110x + Gfx120x + Gfx115x == 1
                      || Gfx103x + Gfx110x + Gfx120x + Gfx115x == 0,
                  "only one of these configs can be chosen.");
    static constexpr architecture value
        = static_cast<bool>(Gfx103x)
              ? architecture::gfx103x
              : (static_cast<bool>(Gfx110x)
                     ? architecture::gfx110x
                     : (static_cast<bool>(Gfx120x)
                            ? architecture::gfx120x
                            : (static_cast<bool>(Gfx115x) ? architecture::gfx115x
                                                          : architecture::unknown)));
};

template <typename HipKernelConfig,
          typename HalfMax,
          typename FltMax,
          typename LaunchDim,
          typename Architecture,
          int Variant,
          int NCHW,
          int MaxN,
          int NElements,
          int N,
          int C,
          int HW,
          int NHW,
          int CHW,
          int Vectorize,
          int VecSize,
          int StashMethod,
          int LoopUnrollMaxN,
          int LoopUnrollMaxHW,
          int LDSGCNSize,
          int LDSSize,
          int UseNodpp>
struct proto_config
{
    static_assert(Vectorize == 0 || Vectorize == 1, "Vectorize must be 0 or 1");
    static_assert(UseNodpp == 0 || UseNodpp == 1, "UseNodpp must be 0 or 1");
    static_assert(NCHW >= 0, "HIP_PLUGIN_BN_NCHW should be always >= 0");
    static_assert(MaxN >= 0, "HIP_PLUGIN_BN_MAXN should be always >= 0");
    static_assert(C >= 0, "HIP_PLUGIN_BN_C should be always >= 0");
    static_assert(N >= 0, "HIP_PLUGIN_BN_N should be always >= 0");
    static_assert(HW >= 0, "HIP_PLUGIN_BN_HW should be always >= 0");
    static_assert(NHW >= 0, "HIP_PLUGIN_BN_NHW should be always >= 0");
    static_assert(CHW >= 0, "HIP_PLUGIN_BN_CHW should be always >= 0");

    static constexpr auto input_type_strategy = HipKernelConfig::input_type_strategy;

    using fp_type = typename std::conditional<
        input_type_strategy == type_strategy::fp16 || input_type_strategy == type_strategy::fpmix,
        _Float16,
        typename std::conditional<input_type_strategy == type_strategy::fp32, float, ushort>::
            type>::type;
    using fp_prec_type = float;
    using fp_accum_type = float;
    static constexpr double epsilon
        = input_type_strategy == type_strategy::fp16 ? 0.0001 : 0.000001;
    static constexpr fp_type max_val
        = input_type_strategy == type_strategy::fp16
              ? HalfMax::value
              : (input_type_strategy == type_strategy::fp32
                     ? FltMax::value
                     : 0); // TODO: not sure if 0 should be the default value of this.
    static constexpr auto launch_dim = LaunchDim{};
    static constexpr unsigned int nchw = static_cast<unsigned int>(NCHW);
    static constexpr unsigned int max_n = static_cast<unsigned int>(MaxN);
    static constexpr unsigned int n_elements = static_cast<unsigned int>(NElements);
    static constexpr unsigned int n = static_cast<unsigned int>(N);
    static constexpr unsigned int c = static_cast<unsigned int>(C);
    static constexpr unsigned int hw = static_cast<unsigned int>(HW);
    static constexpr unsigned int nhw = static_cast<unsigned int>(NHW);
    static constexpr unsigned int chw = static_cast<unsigned int>(CHW);
    static constexpr bool vectorize = static_cast<bool>(Vectorize);
    static constexpr int stash_method = StashMethod;
    static constexpr int loop_unroll_max_n = LoopUnrollMaxN;
    static constexpr int loop_unroll_max_hw = LoopUnrollMaxHW;
    static constexpr unsigned int lds_gcn_size = static_cast<unsigned int>(LDSGCNSize);
    static constexpr unsigned int lds_size = static_cast<unsigned int>(LDSSize);
    static constexpr bool use_nodpp
        = input_type_strategy == type_strategy::fpmix ? false : static_cast<bool>(UseNodpp);
    static constexpr int variant = Variant;
    static constexpr auto target_arch = Architecture::value;
    static constexpr bool use_amdgcn
        = HipKernelConfig::use_amdgcn
          && !(target_arch == architecture::gfx103x || target_arch == architecture::gfx110x
               || target_arch == architecture::gfx120x || target_arch == architecture::gfx115x)
          && !(use_nodpp && (variant != 0));
    static constexpr unsigned int vec_size = vectorize ? VecSize : 1;
    static constexpr unsigned int vec_size_x
        = vectorize && HipKernelConfig::layout_nhwc ? vec_size : 1;
    static constexpr unsigned int vec_size_y
        = vectorize && !HipKernelConfig::layout_nhwc ? vec_size : 1;

    using fp_prec_c_type =
        typename std::conditional<vectorize && HipKernelConfig::layout_nhwc,
                                  typename mapped_vector_type<fp_prec_type, vec_size>::type,
                                  fp_prec_type>::type;

    using fp_prec_ls_type =
        typename std::conditional<vectorize,
                                  typename mapped_vector_type<fp_prec_type, vec_size>::type,
                                  fp_prec_type>::type;

    using fp_c_type =
        typename std::conditional<vectorize && HipKernelConfig::layout_nhwc,
                                  typename mapped_vector_type<fp_type, vec_size>::type,
                                  fp_type>::type;

    using fp_ls_type = typename std::
        conditional<vectorize, typename mapped_vector_type<fp_type, vec_size>::type, fp_type>::type;

    using fp_accum_c_type =
        typename std::conditional<vectorize && HipKernelConfig::layout_nhwc,
                                  typename mapped_vector_type<fp_accum_type, vec_size>::type,
                                  fp_accum_type>::type;

    using fp_accum_ls_type =
        typename std::conditional<vectorize,
                                  typename mapped_vector_type<fp_accum_type, vec_size>::type,
                                  fp_accum_type>::type;
};

} // namespace detail

using config = hip_kernel_provider::batchnorm::detail::proto_config<
    hip_kernel_provider::config,
    hip_kernel_provider::batchnorm::detail::half_max,
    hip_kernel_provider::batchnorm::detail::flt_max,
    hip_kernel_provider::batchnorm::detail::
        launch_dimension<HIP_PLUGIN_BN_GRP0, HIP_PLUGIN_BN_GRP1, HIP_PLUGIN_BN_GRP2>,
    hip_kernel_provider::batchnorm::detail::architecture_switch<HIP_PLUGIN_BN_GFX103X,
                                                                HIP_PLUGIN_BN_GFX110X,
                                                                HIP_PLUGIN_BN_GFX120X,
                                                                HIP_PLUGIN_BN_GFX115X>,
    HIP_PLUGIN_BN_VARIANT,
    HIP_PLUGIN_BN_NCHW,
    HIP_PLUGIN_BN_MAXN,
    HIP_PLUGIN_BN_N_ELEMENTS,
    HIP_PLUGIN_BN_N,
    HIP_PLUGIN_BN_C,
    HIP_PLUGIN_BN_HW,
    HIP_PLUGIN_BN_NHW,
    HIP_PLUGIN_BN_CHW,
    HIP_PLUGIN_BN_VECTORIZE,
    HIP_PLUGIN_BN_VEC_SIZE,
    HIP_PLUGIN_BN_STASH_METHOD,
    HIP_PLUGIN_BN_LOOP_UNROLL_MAXN,
    HIP_PLUGIN_BN_LOOP_UNROLL_MAXHW,
    HIP_PLUGIN_BN_LDSGCN_SIZE,
    HIP_PLUGIN_BN_LDS_SIZE,
    HIP_PLUGIN_BN_NODPP>;

} // namespace batchnorm

} // namespace hip_kernel_provider
