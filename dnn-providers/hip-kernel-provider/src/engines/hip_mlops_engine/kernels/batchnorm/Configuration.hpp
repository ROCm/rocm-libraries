// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "VectorTypes.hpp"
#include <type_traits>

// hip_kernel_provider root configs
namespace hip_kernel_provider
{

enum class TypeStrategy : int
{
    FP16,
    FP32,
    FPMIX,
    BFPMIX,
};

enum class NeuronOpType : int
{
    PASTHRU = 0, // x
    RELU = 3, // max(0, x)
    CLIPPED_RELU = 7, // min(alpha, max(0, x))
    CLAMP = 10,
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
struct ProtoConfig
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

    static constexpr bool LAYOUT_NHWC = static_cast<bool>(LayoutNHWC);
    static constexpr bool SAVE_MEAN_VARIANCE = static_cast<bool>(SaveMeanVariance);
    static constexpr bool RUNNING_RESULT = static_cast<bool>(RunningResult);
    // NOLINTBEGIN(readability-avoid-nested-conditional-operator)
    static constexpr TypeStrategy INPUT_TYPE_STRATEGY
        = (UseFp16 != 0)
              ? TypeStrategy::FP16
              : ((UseFp32 != 0) ? TypeStrategy::FP32
                                : ((UseFpmix != 0) ? TypeStrategy::FPMIX : TypeStrategy::BFPMIX));
    // NOLINTEND(readability-avoid-nested-conditional-operator)
    static constexpr bool USE_AMDGCN = UseAMDGCN != 0;
    static constexpr auto NEURON_OP = static_cast<NeuronOpType>(NrnOpId);
};

} // namespace detail

using config = detail::ProtoConfig<HIP_PLUGIN_LAYOUT_NHWC,
                                   HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE,
                                   HIP_PLUGIN_BN_RUNNING_RESULT,
                                   HIP_PLUGIN_USE_FP16,
                                   HIP_PLUGIN_USE_FP32,
                                   HIP_PLUGIN_USE_FPMIX,
                                   HIP_PLUGIN_USE_BFPMIX,
                                   HIP_PLUGIN_USE_AMDGCN,
                                   HIP_PLUGIN_BN_NRN_OP_ID>;

} // namespace hip_kernel_provider

// hip_kernel_provider batchnorm configs
namespace hip_kernel_provider
{

namespace batchnorm
{

enum class Architecture : int
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
    // NOLINTBEGIN(readability-avoid-nested-conditional-operator)
    static constexpr Architecture VALUE
        = static_cast<bool>(Gfx103x)
              ? Architecture::GFX103X
              : (static_cast<bool>(Gfx110x)
                     ? Architecture::GFX110X
                     : (static_cast<bool>(Gfx120x)
                            ? Architecture::GFX120X
                            : (static_cast<bool>(Gfx115x) ? Architecture::GFX115X
                                                          : Architecture::UNKNOWN)));
    // NOLINTEND(readability-avoid-nested-conditional-operator)
};

template <typename HipKernelConfig,
          typename HalfMax,
          typename FltMax,
          typename Bf16Max,
          typename LaunchDim,
          typename ArchSwitch,
          int Variant,
          int NCHWValue,
          int NElements,
          int NValue,
          int CValue,
          int HWValue,
          int NHWValue,
          int CHWValue,
          int VecSize,
          int StashMethod,
          int LoopUnrollMaxN,
          int LoopUnrollMaxHW,
          int LDSGCNSize,
          int LDSSize,
          int UseNodpp>
struct ProtoConfig
{
    static_assert(UseNodpp == 0 || UseNodpp == 1, "UseNodpp must be 0 or 1");
    static_assert(NCHWValue >= 0, "HIP_PLUGIN_BN_NCHW should be always >= 0");
    static_assert(CValue >= 0, "HIP_PLUGIN_BN_C should be always >= 0");
    static_assert(NValue >= 0, "HIP_PLUGIN_BN_N should be always >= 0");
    static_assert(HWValue >= 0, "HIP_PLUGIN_BN_HW should be always >= 0");
    static_assert(NHWValue >= 0, "HIP_PLUGIN_BN_NHW should be always >= 0");
    static_assert(CHWValue >= 0, "HIP_PLUGIN_BN_CHW should be always >= 0");

    static constexpr auto INPUT_TYPE_STRATEGY = HipKernelConfig::INPUT_TYPE_STRATEGY;

    using fp_type = typename std::conditional_t<
        INPUT_TYPE_STRATEGY == TypeStrategy::FP16 || INPUT_TYPE_STRATEGY == TypeStrategy::FPMIX,
        _Float16,
        typename std::conditional_t<INPUT_TYPE_STRATEGY == TypeStrategy::FP32, float, __bf16>>;
    using fp_prec_type = float;
    using fp_accum_type = float;
    static constexpr double EPSILON = INPUT_TYPE_STRATEGY == TypeStrategy::FP16 ? 0.0001 : 0.000001;
    // NOLINTBEGIN(readability-avoid-nested-conditional-operator)
    static constexpr fp_type MAX_VAL
        = INPUT_TYPE_STRATEGY == TypeStrategy::FP16 || INPUT_TYPE_STRATEGY == TypeStrategy::FPMIX
              ? HalfMax::VALUE
              : (INPUT_TYPE_STRATEGY == TypeStrategy::FP32
                     ? FltMax::VALUE
                     : Bf16Max::
                           VALUE); // According to the old FloatTypes mechanism (on which this is based), for mixed-precision cases the max value is defined as the max of the smaller type
    // NOLINTEND(readability-avoid-nested-conditional-operator)

    static constexpr auto LAUNCH_DIM = LaunchDim{};
    static constexpr unsigned int NCHW = static_cast<unsigned int>(NCHWValue);

    // `max_n` is a limit on the number of batch elements from the x tensor that can be cached in per-thread memory as part of
    // variant 3 kernels. If batchsize (n_elements) exceeds this limit then the kernel doesn't cache the global memory accesses.
    // The constant of 65 is related to the heuristic used in `defaultConfigSpatialSingle` where Batchnorm plans select the kernel
    // variant to use. In BN Fwd training, the heuristics only selects variant 3 when N <= 32, so the caching optimization is always
    // used. In BN Bwd training, the heuristics only selects variant 3 when N > 64, so the caching optimization is never used. If
    // there are future changes to how the heuristic selects variant 3 kernels, then it may be worth revisiting this caching limit.
    static constexpr unsigned int MAX_N = 65;
    static constexpr unsigned int N_ELEMENTS = static_cast<unsigned int>(NElements);
    static constexpr unsigned int N = static_cast<unsigned int>(NValue);
    static constexpr unsigned int C = static_cast<unsigned int>(CValue);
    static constexpr unsigned int HW = static_cast<unsigned int>(HWValue);
    static constexpr unsigned int NHW = static_cast<unsigned int>(NHWValue);
    static constexpr unsigned int CHW = static_cast<unsigned int>(CHWValue);
    static constexpr int STASH_METHOD = StashMethod;
    static constexpr int LOOP_UNROLL_MAX_N = LoopUnrollMaxN;
    static constexpr int LOOP_UNROLL_MAX_HW = LoopUnrollMaxHW;
    static constexpr unsigned int LDS_GCN_SIZE = static_cast<unsigned int>(LDSGCNSize);
    static constexpr unsigned int LDS_SIZE = static_cast<unsigned int>(LDSSize);
    static constexpr bool USE_NODPP
        = INPUT_TYPE_STRATEGY == TypeStrategy::FPMIX ? false : static_cast<bool>(UseNodpp);
    static constexpr int VARIANT = Variant;
    static constexpr auto TARGET_ARCH = ArchSwitch::VALUE;
    static constexpr bool USE_AMDGCN
        = HipKernelConfig::USE_AMDGCN
          && !(TARGET_ARCH == Architecture::GFX103X || TARGET_ARCH == Architecture::GFX110X
               || TARGET_ARCH == Architecture::GFX120X || TARGET_ARCH == Architecture::GFX115X)
          && (!USE_NODPP || (VARIANT == 0));
    static constexpr unsigned int VEC_SIZE = VecSize;
    static constexpr bool VECTORIZE = VecSize > 1;
    static constexpr unsigned int VEC_SIZE_X
        = VECTORIZE && HipKernelConfig::LAYOUT_NHWC ? VEC_SIZE : 1;
    static constexpr unsigned int VEC_SIZE_Y
        = VECTORIZE && !HipKernelConfig::LAYOUT_NHWC ? VEC_SIZE : 1;

    using fp_prec_c_type =
        typename std::conditional<VECTORIZE && HipKernelConfig::LAYOUT_NHWC,
                                  typename MappedVectorType<fp_prec_type, VEC_SIZE>::type,
                                  fp_prec_type>::type;

    using fp_prec_ls_type =
        typename std::conditional<VECTORIZE,
                                  typename MappedVectorType<fp_prec_type, VEC_SIZE>::type,
                                  fp_prec_type>::type;

    using fp_c_type = typename std::conditional<VECTORIZE && HipKernelConfig::LAYOUT_NHWC,
                                                typename MappedVectorType<fp_type, VEC_SIZE>::type,
                                                fp_type>::type;

    using fp_ls_type = typename std::
        conditional<VECTORIZE, typename MappedVectorType<fp_type, VEC_SIZE>::type, fp_type>::type;

    using fp_accum_c_type =
        typename std::conditional<VECTORIZE && HipKernelConfig::LAYOUT_NHWC,
                                  typename MappedVectorType<fp_accum_type, VEC_SIZE>::type,
                                  fp_accum_type>::type;

    using fp_accum_ls_type =
        typename std::conditional<VECTORIZE,
                                  typename MappedVectorType<fp_accum_type, VEC_SIZE>::type,
                                  fp_accum_type>::type;
};

} // namespace detail

using config = hip_kernel_provider::batchnorm::detail::ProtoConfig<
    hip_kernel_provider::config,
    hip_kernel_provider::batchnorm::detail::HalfMax,
    hip_kernel_provider::batchnorm::detail::FltMax,
    hip_kernel_provider::batchnorm::detail::Bf16Max,
    hip_kernel_provider::batchnorm::detail::
        LaunchDimension<HIP_PLUGIN_BN_GRP0, HIP_PLUGIN_BN_GRP1, HIP_PLUGIN_BN_GRP2>,
    hip_kernel_provider::batchnorm::detail::ArchitectureSwitch<HIP_PLUGIN_GFX103X,
                                                               HIP_PLUGIN_GFX110X,
                                                               HIP_PLUGIN_GFX120X,
                                                               HIP_PLUGIN_GFX115X>,
    HIP_PLUGIN_BN_VARIANT,
    HIP_PLUGIN_BN_NCHW,
    HIP_PLUGIN_BN_N_ELEMENTS,
    HIP_PLUGIN_BN_N,
    HIP_PLUGIN_BN_C,
    HIP_PLUGIN_BN_HW,
    HIP_PLUGIN_BN_NHW,
    HIP_PLUGIN_BN_CHW,
    HIP_PLUGIN_BN_VEC_SIZE,
    HIP_PLUGIN_BN_STASH_METHOD,
    HIP_PLUGIN_BN_LOOP_UNROLL_MAXN,
    HIP_PLUGIN_BN_LOOP_UNROLL_MAXHW,
    HIP_PLUGIN_BN_LDSGCN_SIZE,
    HIP_PLUGIN_BN_LDS_SIZE,
    HIP_PLUGIN_BN_NODPP>;

} // namespace batchnorm

} // namespace hip_kernel_provider
