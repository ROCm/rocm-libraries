// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipKernelActivation.hpp"
#include "VectorTypes.hpp"
#include <type_traits>

// hip_kernel_provider root configs
namespace hip_kernel_provider
{

enum class type_strategy : int
{
    fp32,
    fpmix,
    bfpmix,
    fpupcast,
    bfpupcast,
};

namespace detail
{

template <int LayoutChannelLast,
          int SaveMeanVariance,
          int RunningResult,
          int UseFp32,
          int UseFpmix,
          int UseBfpmix,
          int UseFpupcast,
          int UseBfpupcast,
          int UseAMDGCN,
          int NrnOpId>
struct proto_config
{
    static_assert(LayoutChannelLast == 0 || LayoutChannelLast == 1,
                  "LayoutChannelLast (HIP_PLUGIN_LAYOUT_CHANNEL_LAST) must be 0 or 1");
    static_assert(SaveMeanVariance == 0 || SaveMeanVariance == 1,
                  "SaveMeanVariance must be 0 or 1");
    static_assert(RunningResult == 0 || RunningResult == 1, "SaveMeanVariance must be 0 or 1");
    static_assert(UseFp32 == 0 || UseFp32 == 1, "UseFp32 must be 0 or 1");
    static_assert(UseFpmix == 0 || UseFpmix == 1, "UseFpmix must be 0 or 1");
    static_assert(UseBfpmix == 0 || UseBfpmix == 1, "UseBfpmix must be 0 or 1");
    static_assert(UseFpupcast == 0 || UseFpupcast == 1, "UseFpupcast must be 0 or 1");
    static_assert(UseBfpupcast == 0 || UseBfpupcast == 1, "UseBfpupcast must be 0 or 1");
    static_assert(UseFp32 + UseFpmix + UseBfpmix + UseFpupcast + UseBfpupcast == 1,
                  "only one of these configs can and must be chosen.");
    static_assert(UseAMDGCN == 0 || UseAMDGCN == 1, "UseAMDGCN must be 0 or 1");
    static_assert(NrnOpId >= 0 && NrnOpId <= 10,
                  "NrnOpId can only be integer between 0-10 (inclusive)");

    static constexpr bool layout_channel_last = static_cast<bool>(LayoutChannelLast);
    static constexpr bool save_mean_variance = static_cast<bool>(SaveMeanVariance);
    static constexpr bool running_result = static_cast<bool>(RunningResult);
    static constexpr type_strategy input_type_strategy
        = UseFp32 ? type_strategy::fp32
                  : (UseFpmix ? type_strategy::fpmix
                              : (UseBfpmix ? type_strategy::bfpmix
                                           : (UseFpupcast ? type_strategy::fpupcast
                                                          : type_strategy::bfpupcast)));
    static constexpr bool use_amdgcn = UseAMDGCN;
    static constexpr auto neuron_op = static_cast<ActivationMode>(NrnOpId);
};

} // namespace detail

using config = detail::proto_config<HIP_PLUGIN_LAYOUT_CHANNEL_LAST,
                                    HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE,
                                    HIP_PLUGIN_BN_RUNNING_RESULT,
                                    HIP_PLUGIN_USE_FP32,
                                    HIP_PLUGIN_USE_FPMIX,
                                    HIP_PLUGIN_USE_BFPMIX,
                                    HIP_PLUGIN_USE_FPUPCAST,
                                    HIP_PLUGIN_USE_BFPUPCAST,
                                    HIP_PLUGIN_USE_AMDGCN,
                                    HIP_PLUGIN_BN_NRN_OP_ID>;

} // namespace hip_kernel_provider

// hip_kernel_provider batchnorm configs
namespace hip_kernel_provider::batchnorm
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

// TODO: why this is here, because before c++ 20, double is not supported to be template parameter
struct half_max
{
    static constexpr double value = 65504;
};

// TODO: why this is here, because before c++ 20, double is not supported to be template parameter
struct flt_max
{
    static constexpr double value = 3.402823466e+38;
};

struct bf16_max
{
    static constexpr double value = 0x1.fep+127;
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
    static constexpr unsigned int bounds = grp0 * grp1 * grp2;
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
          typename InputType,
          typename OutputType,
          typename ScaleBiasType,
          typename MeanInvVarianceType,
          typename HalfMax,
          typename FltMax,
          typename Bf16Max,
          typename LaunchDim,
          typename LaunchDimFinal,
          typename LaunchGroups,
          typename Architecture,
          int Variant,
          int BatchElements,
          int Batches,
          int Channels,
          int SpatialSize,
          int VecSize,
          int StashMethod,
          int LoopUnrollMaxBatch,
          int LoopUnrollMaxSpatial,
          int LDSGCNSize,
          int LDSSize,
          int UseNodpp,
          int UseSaved>
struct proto_config
{
    static_assert(UseNodpp == 0 || UseNodpp == 1, "UseNodpp must be 0 or 1");
    static_assert(Batches >= 0, "HIP_PLUGIN_BN_BATCHES should be always >= 0");
    static_assert(Channels >= 0, "HIP_PLUGIN_BN_CHANNELS should be always >= 0");
    static_assert(SpatialSize >= 0, "HIP_PLUGIN_BN_SPATIAL_SIZE should be always >= 0");

    static constexpr auto input_type_strategy = HipKernelConfig::input_type_strategy;

    using input_type = InputType;
    using output_type = OutputType;
    using scale_bias_type = ScaleBiasType;
    using mean_inv_variance_type = MeanInvVarianceType;
    using compute_type = float;

    using input_vec_type =
        typename hip_kernel_provider::mapped_vector_type<input_type, VecSize>::type;
    using output_vec_type =
        typename hip_kernel_provider::mapped_vector_type<output_type, VecSize>::type;
    using scale_bias_vec_type =
        typename hip_kernel_provider::mapped_vector_type<scale_bias_type, VecSize>::type;
    using mean_inv_variance_vec_type =
        typename hip_kernel_provider::mapped_vector_type<mean_inv_variance_type, VecSize>::type;
    using compute_vec_type =
        typename hip_kernel_provider::mapped_vector_type<compute_type, VecSize>::type;

    using input_channel_vec_type = typename std::
        conditional<HipKernelConfig::layout_channel_last, input_vec_type, input_type>::type;
    using output_channel_vec_type = typename std::
        conditional<HipKernelConfig::layout_channel_last, output_vec_type, output_type>::type;
    using scale_bias_channel_vec_type =
        typename std::conditional<HipKernelConfig::layout_channel_last,
                                  scale_bias_vec_type,
                                  scale_bias_type>::type;
    using mean_inv_variance_channel_vec_type =
        typename std::conditional<HipKernelConfig::layout_channel_last,
                                  mean_inv_variance_vec_type,
                                  mean_inv_variance_type>::type;
    using compute_channel_vec_type = typename std::
        conditional<HipKernelConfig::layout_channel_last, compute_vec_type, compute_type>::type;

    using fp_type = typename std::conditional<
        input_type_strategy == type_strategy::fpmix,
        _Float16,
        typename std::conditional<input_type_strategy == type_strategy::fp32, float, __bf16>::
            type>::type;
    using fp_prec_type = float;
    using fp_accum_type = float;
    static constexpr fp_type max_val
        = input_type_strategy == type_strategy::fpmix
              ? HalfMax::value
              : (input_type_strategy == type_strategy::fp32
                     ? FltMax::value
                     : Bf16Max::
                           value); // According to the old FloatTypes mechanism (on which this is based), for mixed-precision cases the max value is defined as the max of the smaller type
    static constexpr auto launch_dim = LaunchDim{};
    static constexpr auto launch_dim_final = LaunchDimFinal{};
    static constexpr auto launch_groups = LaunchGroups{};
    static constexpr unsigned int tensor_size
        = static_cast<unsigned int>(Batches * Channels * SpatialSize);

    // `max_batches` is a limit on the number of batch elements from the x tensor that can be cached in per-thread memory as part of
    // variant 3 kernels. If batchsize (batch_elements) exceeds this limit then the kernel doesn't cache the global memory accesses.
    // The constant of 65 is related to the heuristic used in `defaultConfigSpatialSingle` where Batchnorm plans select the kernel
    // variant to use. In BN Fwd training, the heuristics only selects variant 3 when N <= 32, so the caching optimization is always
    // used. In BN Bwd training, the heuristics only selects variant 3 when N > 64, so the caching optimization is never used. If
    // there are future changes to how the heuristic selects variant 3 kernels, then it may be worth revisiting this caching limit.
    static constexpr unsigned int max_batches = 65;
    static constexpr unsigned int batch_elements = static_cast<unsigned int>(BatchElements);
    static constexpr unsigned int batches = static_cast<unsigned int>(Batches);
    static constexpr unsigned int channels = static_cast<unsigned int>(Channels);
    static constexpr unsigned int spatial_size = static_cast<unsigned int>(SpatialSize);
    static constexpr unsigned int channel_slice_size
        = static_cast<unsigned int>(Batches * SpatialSize);
    static constexpr float i_channel_slice_size = 1.0f / static_cast<float>(channel_slice_size);
    static constexpr unsigned int batch_size = static_cast<unsigned int>(Channels * SpatialSize);
    static constexpr int stash_method = StashMethod;
    static constexpr int loop_unroll_max_batch = LoopUnrollMaxBatch;
    static constexpr int loop_unroll_max_spatial = LoopUnrollMaxSpatial;
    static constexpr unsigned int lds_gcn_size = static_cast<unsigned int>(LDSGCNSize);
    static constexpr unsigned int lds_size = static_cast<unsigned int>(LDSSize);
    static constexpr bool use_nodpp
        = input_type_strategy == type_strategy::fpmix ? false : static_cast<bool>(UseNodpp);
    static constexpr bool use_saved = static_cast<bool>(UseSaved);
    static constexpr int variant = Variant;
    static constexpr auto target_arch = Architecture::value;
    static constexpr bool use_amdgcn
        = HipKernelConfig::use_amdgcn
          && !(target_arch == architecture::gfx103x || target_arch == architecture::gfx110x
               || target_arch == architecture::gfx120x || target_arch == architecture::gfx115x)
          && !(use_nodpp && (variant != 0));
    static constexpr unsigned int vec_size = VecSize;
    static constexpr unsigned int vec_size_x = HipKernelConfig::layout_channel_last ? vec_size : 1;
    static constexpr unsigned int vec_size_y = !HipKernelConfig::layout_channel_last ? vec_size : 1;
};

} // namespace detail

using config = hip_kernel_provider::batchnorm::detail::proto_config<
    hip_kernel_provider::config,
    HIP_PLUGIN_BN_INPUT_TYPE,
    HIP_PLUGIN_BN_OUTPUT_TYPE,
    HIP_PLUGIN_BN_SCALE_TYPE,
    HIP_PLUGIN_BN_MEAN_VAR_TYPE,
    hip_kernel_provider::batchnorm::detail::half_max,
    hip_kernel_provider::batchnorm::detail::flt_max,
    hip_kernel_provider::batchnorm::detail::bf16_max,
    hip_kernel_provider::batchnorm::detail::
        launch_dimension<HIP_PLUGIN_BN_GRP0, HIP_PLUGIN_BN_GRP1, HIP_PLUGIN_BN_GRP2>,
    hip_kernel_provider::batchnorm::detail::launch_dimension<HIP_PLUGIN_BN_GRP0_FINAL,
                                                             HIP_PLUGIN_BN_GRP1_FINAL,
                                                             HIP_PLUGIN_BN_GRP2_FINAL>,
    hip_kernel_provider::batchnorm::detail::
        launch_dimension<HIP_PLUGIN_BN_GRPS0, HIP_PLUGIN_BN_GRPS1, HIP_PLUGIN_BN_GRPS2>,
    hip_kernel_provider::batchnorm::detail::architecture_switch<HIP_PLUGIN_GFX103X,
                                                                HIP_PLUGIN_GFX110X,
                                                                HIP_PLUGIN_GFX120X,
                                                                HIP_PLUGIN_GFX115X>,
    HIP_PLUGIN_BN_VARIANT,
    HIP_PLUGIN_BN_BATCH_ELEMENTS,
    HIP_PLUGIN_BN_BATCHES,
    HIP_PLUGIN_BN_CHANNELS,
    HIP_PLUGIN_BN_SPATIAL_SIZE,
    HIP_PLUGIN_BN_VEC_SIZE,
    HIP_PLUGIN_BN_STASH_METHOD,
    HIP_PLUGIN_BN_LOOP_UNROLL_MAX_BATCH,
    HIP_PLUGIN_BN_LOOP_UNROLL_MAX_SPATIAL,
    HIP_PLUGIN_BN_LDSGCN_SIZE,
    HIP_PLUGIN_BN_LDS_SIZE,
    HIP_PLUGIN_BN_NODPP,
    HIP_PLUGIN_BN_USESAVED>;

} // namespace hip_kernel_provider::batchnorm
