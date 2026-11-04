// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipKernelActivation.hpp"
#include "VectorTypes.hpp"
#include <type_traits>

// hip_kernel_provider root configs
namespace hip_kernel_provider
{

enum class TypeStrategy : int
{
    FP32,
    FPMIX,
    BFPMIX,
    FPUPCAST,
    BFPUPCAST,
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
struct ProtoConfig
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

    static constexpr bool LAYOUT_CHANNEL_LAST = static_cast<bool>(LayoutChannelLast);
    static constexpr bool SAVE_MEAN_VARIANCE = static_cast<bool>(SaveMeanVariance);
    static constexpr bool RUNNING_RESULT = static_cast<bool>(RunningResult);
    static constexpr TypeStrategy INPUT_TYPE_STRATEGY
        = (UseFp32 != 0)
              ? TypeStrategy::FP32
              : ((UseFpmix != 0) // NOLINT(readability-avoid-nested-conditional-operator)
                     ? TypeStrategy::FPMIX // NOLINT(readability-avoid-nested-conditional-operator)
                     : ((UseBfpmix != 0) // NOLINT(readability-avoid-nested-conditional-operator)
                            ? TypeStrategy::
                                  BFPMIX // NOLINT(readability-avoid-nested-conditional-operator)
                            : ((UseFpupcast // NOLINT(readability-avoid-nested-conditional-operator)
                                != 0) // NOLINT(readability-avoid-nested-conditional-operator)
                                   ? TypeStrategy::
                                         FPUPCAST // NOLINT(readability-avoid-nested-conditional-operator)
                                   : TypeStrategy::BFPUPCAST)));
    static constexpr bool USE_AMDGCN = UseAMDGCN != 0;
    static constexpr auto NEURON_OP = static_cast<ActivationMode>(NrnOpId);
};

} // namespace detail

using config = detail::ProtoConfig<HIP_PLUGIN_LAYOUT_CHANNEL_LAST,
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

enum class GpuArchitecture : int
{
    UNKNOWN,
    GFX103X,
    GFX110X,
    GFX115X,
    GFX120X,
};

namespace detail
{

// TODO: why this is here, because before c++ 20, double is not supported to be template parameter
struct HalfMax
{
    static constexpr double VALUE = 65504;
};

// TODO: why this is here, because before c++ 20, double is not supported to be template parameter
struct FltMax
{
    static constexpr double VALUE = 3.402823466e+38;
};

struct Bf16Max
{
    static constexpr double VALUE = 0x1.fep+127;
};

template <int Grp0, int Grp1, int Grp2>
struct LaunchDimension
{
    static_assert(Grp0 >= 0, "HIP_PLUGIN_BN_GRP0 should be always >= 0");
    static_assert(Grp1 >= 0, "HIP_PLUGIN_BN_GRP1 should be always >= 0");
    static_assert(Grp2 >= 0, "HIP_PLUGIN_BN_GRP2 should be always >= 0");
    static constexpr unsigned int GRP0 = static_cast<unsigned int>(Grp0);
    static constexpr unsigned int GRP1 = static_cast<unsigned int>(Grp1);
    static constexpr unsigned int GRP2 = static_cast<unsigned int>(Grp2);
    static constexpr unsigned int BOUNDS = GRP0 * GRP1 * GRP2;
};

template <int Gfx103x, int Gfx110x, int Gfx120x, int Gfx115x>
struct ArchitectureSwitch
{
    static_assert(Gfx103x == 0 || Gfx103x == 1, "Gfx103x must be 0 or 1");
    static_assert(Gfx110x == 0 || Gfx110x == 1, "Gfx110x must be 0 or 1");
    static_assert(Gfx120x == 0 || Gfx120x == 1, "Gfx120x must be 0 or 1");
    static_assert(Gfx115x == 0 || Gfx115x == 1, "Gfx115x must be 0 or 1");
    static_assert(Gfx103x + Gfx110x + Gfx120x + Gfx115x == 1
                      || Gfx103x + Gfx110x + Gfx120x + Gfx115x == 0,
                  "only one of these configs can be chosen.");
    static constexpr GpuArchitecture VALUE
        = static_cast<bool>(Gfx103x)
              ? GpuArchitecture::GFX103X
              : (static_cast<bool>(Gfx110x) // NOLINT(readability-avoid-nested-conditional-operator)
                     ? GpuArchitecture::GFX110X
                     : (static_cast<bool>( // NOLINT(readability-avoid-nested-conditional-operator)
                            Gfx120x) // NOLINT(readability-avoid-nested-conditional-operator)
                            ? GpuArchitecture::GFX120X
                            : (static_cast< // NOLINT(readability-avoid-nested-conditional-operator)
                                   bool>( // NOLINT(readability-avoid-nested-conditional-operator)
                                   Gfx115x) // NOLINT(readability-avoid-nested-conditional-operator)
                                   ? GpuArchitecture::
                                         GFX115X // NOLINT(readability-avoid-nested-conditional-operator)
                                   : GpuArchitecture::UNKNOWN)));
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
struct ProtoConfig
{
    static_assert(UseNodpp == 0 || UseNodpp == 1, "UseNodpp must be 0 or 1");
    static_assert(Batches >= 0, "HIP_PLUGIN_BN_BATCHES should be always >= 0");
    static_assert(Channels >= 0, "HIP_PLUGIN_BN_CHANNELS should be always >= 0");
    static_assert(SpatialSize >= 0, "HIP_PLUGIN_BN_SPATIAL_SIZE should be always >= 0");

    static constexpr auto INPUT_TYPE_STRATEGY = HipKernelConfig::INPUT_TYPE_STRATEGY;

    using input_type = InputType;
    using output_type = OutputType;
    using scale_bias_type = ScaleBiasType;
    using mean_inv_variance_type = MeanInvVarianceType;
    using compute_type = float;

    using input_vec_type =
        typename hip_kernel_provider::MappedVectorType<input_type, VecSize>::type;
    using output_vec_type =
        typename hip_kernel_provider::MappedVectorType<output_type, VecSize>::type;
    using scale_bias_vec_type =
        typename hip_kernel_provider::MappedVectorType<scale_bias_type, VecSize>::type;
    using mean_inv_variance_vec_type =
        typename hip_kernel_provider::MappedVectorType<mean_inv_variance_type, VecSize>::type;
    using compute_vec_type =
        typename hip_kernel_provider::MappedVectorType<compute_type, VecSize>::type;

    using input_channel_vec_type
        = std::conditional_t<HipKernelConfig::LAYOUT_CHANNEL_LAST, input_vec_type, input_type>;
    using output_channel_vec_type
        = std::conditional_t<HipKernelConfig::LAYOUT_CHANNEL_LAST, output_vec_type, output_type>;
    using scale_bias_channel_vec_type = std::
        conditional_t<HipKernelConfig::LAYOUT_CHANNEL_LAST, scale_bias_vec_type, scale_bias_type>;
    using mean_inv_variance_channel_vec_type
        = std::conditional_t<HipKernelConfig::LAYOUT_CHANNEL_LAST,
                             mean_inv_variance_vec_type,
                             mean_inv_variance_type>;
    using compute_channel_vec_type
        = std::conditional_t<HipKernelConfig::LAYOUT_CHANNEL_LAST, compute_vec_type, compute_type>;

    using fp_type = std::conditional_t<
        INPUT_TYPE_STRATEGY == TypeStrategy::FPMIX,
        _Float16,
        std::conditional_t<INPUT_TYPE_STRATEGY == TypeStrategy::FP32, float, __bf16>>;
    using fp_prec_type = float;
    using fp_accum_type = float;
    static constexpr fp_type MAX_VAL
        = INPUT_TYPE_STRATEGY == TypeStrategy::FPMIX
              ? HalfMax::value
              : (INPUT_TYPE_STRATEGY // NOLINT(readability-avoid-nested-conditional-operator)
                         == TypeStrategy::
                             FP32 // NOLINT(readability-avoid-nested-conditional-operator)
                     ? FltMax::value
                     : Bf16Max::
                           value); // According to the old FloatTypes mechanism (on which this is based), for mixed-precision cases the max value is defined as the max of the smaller type
    static constexpr auto LAUNCH_DIM = LaunchDim{};
    static constexpr auto LAUNCH_DIM_FINAL = LaunchDimFinal{};
    static constexpr auto LAUNCH_GROUPS = LaunchGroups{};
    static constexpr unsigned int TENSOR_SIZE
        = static_cast<unsigned int>(Batches * Channels * SpatialSize);

    // `max_batches` is a limit on the number of batch elements from the x tensor that can be cached in per-thread memory as part of
    // variant 3 kernels. If batchsize (batch_elements) exceeds this limit then the kernel doesn't cache the global memory accesses.
    // The constant of 65 is related to the heuristic used in `defaultConfigSpatialSingle` where Batchnorm plans select the kernel
    // variant to use. In BN Fwd training, the heuristics only selects variant 3 when N <= 32, so the caching optimization is always
    // used. In BN Bwd training, the heuristics only selects variant 3 when N > 64, so the caching optimization is never used. If
    // there are future changes to how the heuristic selects variant 3 kernels, then it may be worth revisiting this caching limit.
    static constexpr unsigned int MAX_BATCHES = 65;
    static constexpr unsigned int BATCH_ELEMENTS = static_cast<unsigned int>(BatchElements);
    static constexpr unsigned int BATCHES = static_cast<unsigned int>(Batches);
    static constexpr unsigned int CHANNELS = static_cast<unsigned int>(Channels);
    static constexpr unsigned int SPATIAL_SIZE = static_cast<unsigned int>(SpatialSize);
    static constexpr unsigned int CHANNEL_SLICE_SIZE
        = static_cast<unsigned int>(Batches * SpatialSize);
    static constexpr float I_CHANNEL_SLICE_SIZE = 1.0f / static_cast<float>(CHANNEL_SLICE_SIZE);
    static constexpr unsigned int BATCH_SIZE = static_cast<unsigned int>(Channels * SpatialSize);
    static constexpr int STASH_METHOD = StashMethod;
    static constexpr int LOOP_UNROLL_MAX_BATCH = LoopUnrollMaxBatch;
    static constexpr int LOOP_UNROLL_MAX_SPATIAL = LoopUnrollMaxSpatial;
    static constexpr unsigned int LDS_GCN_SIZE = static_cast<unsigned int>(LDSGCNSize);
    static constexpr unsigned int LDS_SIZE = static_cast<unsigned int>(LDSSize);
    static constexpr bool USE_NODPP
        = INPUT_TYPE_STRATEGY == TypeStrategy::FPMIX ? false : static_cast<bool>(UseNodpp);
    static constexpr bool USE_SAVED = static_cast<bool>(UseSaved);
    static constexpr int VARIANT = Variant;
    static constexpr auto TARGET_ARCH = Architecture::VALUE;
    static constexpr bool USE_AMDGCN
        = HipKernelConfig::USE_AMDGCN
          && !(TARGET_ARCH == GpuArchitecture::GFX103X || TARGET_ARCH == GpuArchitecture::GFX110X
               || TARGET_ARCH == GpuArchitecture::GFX120X
               || TARGET_ARCH == GpuArchitecture::GFX115X)
          && (!USE_NODPP || (VARIANT == 0));
    static constexpr unsigned int VEC_SIZE = VecSize;
    static constexpr unsigned int VEC_SIZE_X = HipKernelConfig::LAYOUT_CHANNEL_LAST ? VEC_SIZE : 1;
    static constexpr unsigned int VEC_SIZE_Y = !HipKernelConfig::LAYOUT_CHANNEL_LAST ? VEC_SIZE : 1;
};

} // namespace detail

using config = hip_kernel_provider::batchnorm::detail::ProtoConfig<
    hip_kernel_provider::config,
    HIP_PLUGIN_BN_INPUT_TYPE,
    HIP_PLUGIN_BN_OUTPUT_TYPE,
    HIP_PLUGIN_BN_SCALE_TYPE,
    HIP_PLUGIN_BN_MEAN_VAR_TYPE,
    hip_kernel_provider::batchnorm::detail::HalfMax,
    hip_kernel_provider::batchnorm::detail::FltMax,
    hip_kernel_provider::batchnorm::detail::Bf16Max,
    hip_kernel_provider::batchnorm::detail::
        LaunchDimension<HIP_PLUGIN_BN_GRP0, HIP_PLUGIN_BN_GRP1, HIP_PLUGIN_BN_GRP2>,
    hip_kernel_provider::batchnorm::detail::LaunchDimension<HIP_PLUGIN_BN_GRP0_FINAL,
                                                            HIP_PLUGIN_BN_GRP1_FINAL,
                                                            HIP_PLUGIN_BN_GRP2_FINAL>,
    hip_kernel_provider::batchnorm::detail::
        LaunchDimension<HIP_PLUGIN_BN_GRPS0, HIP_PLUGIN_BN_GRPS1, HIP_PLUGIN_BN_GRPS2>,
    hip_kernel_provider::batchnorm::detail::ArchitectureSwitch<HIP_PLUGIN_GFX103X,
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
