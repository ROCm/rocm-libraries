// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#define CONV_WINO_RAGE_RXS_CPP

#include <cstdint>
#include <cstdlib>
#include <miopen/kernel_build_params.hpp>
#include <miopen/conv/invokers/gcn_asm_wino.hpp>
#include <miopen/conv/kernel_interface/winograd_kernel_interface.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/utils.hpp>
#include <miopen/stringutils.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3)

namespace miopen {

namespace solver {

using ProblemDescription = miopen::conv::ProblemDescription;
using WinoShaderArgs     = miopen::WinoShaderArgsV2;

namespace {

struct PerfModelResult
{
    float wti;
    float granularity_loss;
    uint32_t n_groups;
    uint64_t predicted_clk;
};

struct PerfModelCost
{
    uint64_t const_cost;
    uint64_t fe_cost;
    uint64_t ph_cost;
    uint64_t be_cost;
};

struct PerfModelParams
{
    uint64_t mac_rate;  // MAC operations per CU per clock
    PerfModelCost cost; // Cost of internal shader operations
};

class ShaderModel
{
protected:
    const WinoShaderArgsV2 args;
    const uint32_t cu_count;
    PerfModelParams model_params;

public:
    ShaderModel(const WinoShaderArgsV2& shader_args, uint32_t cu_cnt)
        : args(shader_args), cu_count(cu_cnt)
    {
    }

    virtual ~ShaderModel() = default;

    PerfModelResult ComputeWti(bool is_fp32) const
    {
        PerfModelResult out{};
        constexpr uint32_t max_dispatches = 8;

        for(uint32_t n_dispatches = 1; n_dispatches <= max_dispatches; n_dispatches++)
        {
            const uint64_t n_groups = static_cast<uint64_t>(cu_count) * n_dispatches;
            if(!IsShaderConstraintsMet(n_groups))
                continue;

            PerfModelResult prediction = PerfPrediction(n_groups, is_fp32);

            if(prediction.wti > out.wti)
                out = prediction;
        }

        return out;
    }

    virtual bool IsShaderConstraintsMet(const uint64_t n_groups) const
    {
        // clang-format off
        return args.dimsFit16bit()
            && args.batchTensorSizesFit31bits()
            && args.paddedSizesFit16bits()
            && n_groups < WinoShaderArgsV2::PowOf2<16>()
            && DivCeil(args.Kg, 32) <= n_groups;
        // clang-format on
    }

protected:
    // Divide two non-negative integers and return ceil of the quotient
    uint64_t DivCeil(uint64_t numer, uint64_t denom) const { return (numer + denom - 1) / denom; }
    uint64_t RoundUpToMultiple(uint64_t val, uint64_t mul) const { return DivCeil(val, mul) * mul; }
    PerfModelResult PerfPrediction(const uint64_t n_groups, bool is_fp32) const
    {
        PerfModelResult out{};

        constexpr uint64_t t_R  = 3;
        constexpr uint64_t t_S  = 3;
        constexpr uint64_t t_oH = 2;
        constexpr uint64_t t_oW = 2;

        constexpr uint64_t nhw_factor = 62;
        constexpr uint64_t k_factor   = 32;
        const uint64_t c_factor       = is_fp32 ? 8 : 16;
        const uint64_t nhw_factor_g   = RoundUpToMultiple(nhw_factor, 32);
        const uint64_t nkhw_per_work  = k_factor * nhw_factor_g * t_oH * t_oW;

        const uint64_t Rg  = RoundUpToMultiple(args.R, t_R);
        const uint64_t Sg  = RoundUpToMultiple(args.S, t_S);
        const uint64_t Cg  = RoundUpToMultiple(args.Cg, c_factor);
        const uint64_t Kg  = RoundUpToMultiple(args.Kg, k_factor);
        const uint64_t oHg = RoundUpToMultiple(args.out_h, t_oH);
        const uint64_t oWg = RoundUpToMultiple(args.out_w, t_oW) + t_oW;

        const uint64_t s_loops = Sg / t_S;
        const uint64_t r_loops = Rg / t_R;
        const uint64_t c_loops = Cg / c_factor;
        const uint64_t k_tiles = Kg / k_factor;

        const uint64_t nhw_tiles = args.N * DivCeil(oHg, t_oH) * DivCeil(oWg, t_oW);

        const uint64_t n_groups_e   = k_tiles * (n_groups / k_tiles);
        const uint64_t n_dispatches = DivCeil(args.G * n_groups_e, cu_count);

        const uint64_t n_works        = k_tiles * DivCeil(nhw_tiles, nhw_factor);
        const uint64_t n_works_per_cu = DivCeil(n_works, n_groups_e);

        const uint64_t macsg =
            n_dispatches * n_works_per_cu * cu_count * nkhw_per_work * nhw_factor_g * Cg * Rg * Sg;
        const uint64_t macs = static_cast<uint64_t>(args.N) * args.G * Kg * Cg * args.out_h *
                              args.R * args.out_w * args.S;

        out.granularity_loss = static_cast<float>(macsg - macs) / macsg;

        const uint64_t n_consts = n_dispatches;
        const uint64_t fe_calls = n_dispatches * n_works_per_cu * s_loops * r_loops;
        const uint64_t ph_calls = n_dispatches * n_works_per_cu * s_loops * r_loops * c_loops;
        const uint64_t be_calls = n_dispatches * n_works_per_cu;

        const uint64_t const_cost = model_params.cost.const_cost;
        const uint64_t fe_cost    = model_params.cost.fe_cost;
        const uint64_t ph_cost    = model_params.cost.ph_cost;
        const uint64_t be_cost    = model_params.cost.be_cost;

        out.predicted_clk =
            n_consts * const_cost + fe_calls * fe_cost + ph_calls * ph_cost + be_calls * be_cost;

        const float ideal_direct_clk = static_cast<float>(macs) / model_params.mac_rate / cu_count;

        out.wti      = ideal_direct_clk / out.predicted_clk;
        out.n_groups = n_groups;

        return out;
    }
};

class ShaderModelV4_6 : public ShaderModel
{
public:
    ShaderModelV4_6(const WinoShaderArgsV2& shader_args,
                    uint32_t cu_cnt,
                    const PerfModelParams& perf_params)
        : ShaderModel(shader_args, cu_cnt)
    {
        model_params = perf_params;
    }

    bool IsShaderConstraintsMet(const uint64_t n_groups) const override
    {
        return ShaderModel::IsShaderConstraintsMet(n_groups) && args.R_S_fit3x3();
    }
};

class ShaderModelV4_9 : public ShaderModel
{
public:
    ShaderModelV4_9(const WinoShaderArgsV2& shader_args,
                    uint32_t cu_cnt,
                    const PerfModelParams& perf_params)
        : ShaderModel(shader_args, cu_cnt)
    {
        model_params = perf_params;
    }
};

class ShaderModelFactory
{
public:
    // We keep two kernel versions because V4_6 delivers better performance
    // on both gfx12x(+4%) and gfx942(+6%) for cases where input channels <= 16.
    enum class KernelVersion
    {
        V4_6, // 3x3 filters, FP16 only
        V4_9  // supports 3x3 and other filter sizes, FP16/FP32/BF16
    };

    // Performance model parameters for different hardware architectures and data types
    // Format: {mac_rate, {const_cost, fe_cost, ph_cost, be_cost}}
    struct PerfParams
    {
        // clang-format off
        static constexpr PerfModelParams GFX942_V4_9_fp16{1024, {22724, 512, 1372, 2244}};
        static constexpr PerfModelParams GFX942_V4_9_bf16{1024, {22724, 512, 1660, 2656}};
        static constexpr PerfModelParams GFX942_V4_9_fp32{128, {26044, 512, 2468, 2504}};
        static constexpr PerfModelParams GFX942_V4_6{1024, {22850, 244, 1396, 2244}};

        static constexpr PerfModelParams GFX12_V4_9{512, {9740, 182, 1506, 1533}};
        static constexpr PerfModelParams GFX12_V4_6{512, {9505, 79, 1522, 1533}};
        // clang-format on
    };

    static KernelVersion DetermineKernelVersion(const WinoShaderArgsV2& args, bool is_fp16)
    {
        if(args.R_S_fit3x3() && is_fp16)
        {
            return KernelVersion::V4_6;
        }
        else
        {
            return KernelVersion::V4_9;
        }
    }

    // Get performance parameters for specific hardware
    static PerfModelParams GetPerfParams(const std::string& dev_name,
                                         KernelVersion kernel_version,
                                         const ProblemDescription& problem)
    {
        if(StartsWith(dev_name, "gfx942"))
        {
            return (kernel_version == KernelVersion::V4_6) ? PerfParams::GFX942_V4_6
                   : (problem.IsFp16())                    ? PerfParams::GFX942_V4_9_fp16
                   : (problem.IsBfp16())                   ? PerfParams::GFX942_V4_9_bf16
                                                           : PerfParams::GFX942_V4_9_fp32;
        }
        else if(StartsWith(dev_name, "gfx12"))
        {
            return (kernel_version == KernelVersion::V4_6) ? PerfParams::GFX12_V4_6
                                                           : PerfParams::GFX12_V4_9;
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError, "Unsupported device architecture: " + dev_name);
        }
    }

    static std::unique_ptr<ShaderModel> Create(const std::string& dev_name,
                                               const WinoShaderArgsV2& args,
                                               uint32_t cu_count,
                                               const ProblemDescription& problem,
                                               KernelVersion kernel_version)
    {
        auto perf_params = GetPerfParams(dev_name, kernel_version, problem);

        if(kernel_version == KernelVersion::V4_6)
        {
            return std::make_unique<ShaderModelV4_6>(args, cu_count, perf_params);
        }
        else // V4_9
        {
            return std::make_unique<ShaderModelV4_9>(args, cu_count, perf_params);
        }
    }

    static std::unique_ptr<ShaderModel> Create(const std::string& dev_name,
                                               const WinoShaderArgsV2& args,
                                               uint32_t cu_count,
                                               const ProblemDescription& problem)
    {
        auto kernel_version = DetermineKernelVersion(args, problem.IsFp16());
        return Create(dev_name, args, cu_count, problem, kernel_version);
    }
};

template <uint32_t Winodata, uint32_t Winofilter>
struct ConvWinoRageRxSCommon
{
    static bool IsApplicable(const ExecutionContext&, const ProblemDescription&);
    static float GetWti(const ExecutionContext&, const ProblemDescription&);
    static ConvSolution GetSolution(const ExecutionContext&,
                                    const ProblemDescription&,
                                    bool fused                        = false,
                                    bool do_bias                      = false,
                                    miopenActivationMode_t activ_mode = miopenActivationPASTHRU);

private:
    static int64_t getMaxNGroups(const ExecutionContext& ctx)
    {
        // return max number of groups that ShaderModel considers
        // see ShaderMode::ComputeWti()
        return ctx.GetStream().GetMaxHardwareComputeUnits() * 8;
    }
};

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoRageRxSCommon<Winodata, Winofilter>::IsApplicable(const ExecutionContext& ctx,
                                                               const ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_AMD_WINOGRAD_RAGE_RXS_F2X3))
        return false;
    if(!ctx.use_asm_kernels)
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(problem.HasNonPackedTensors())
        return false;

    WinoShaderArgs args;
    if(!args.SetConvParams(problem))
        return false;

    const auto devName = ctx.GetStream().GetDeviceName();
    if(devName == "gfx942")
    {
        if(!(problem.IsFp16() || problem.IsFp32() || problem.IsBfp16()))
            return false;
    }
    else if(StartsWith(devName, "gfx12"))
    {
        if(!(problem.IsFp16()))
            return false;
    }
    else
    {
        return false;
    }

    const auto& target = ctx.GetStream().GetTargetProperties();
    if(target.isXnackEnabled())
        return false;

    if(!(problem.GetKernelStrideH() == 1 && problem.GetKernelStrideW() == 1))
        return false;
    if(!(problem.GetDilationH() == 1 && problem.GetDilationW() == 1))
        return false;

    args.n_groups     = getMaxNGroups(ctx);
    auto shader_model = ShaderModelFactory::Create(
        devName, args, ctx.GetStream().GetMaxHardwareComputeUnits(), problem);

    return shader_model->IsShaderConstraintsMet(args.n_groups);
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoRageRxSCommon<Winodata, Winofilter>::GetWti(const ExecutionContext& ctx,
                                                          const ProblemDescription& problem)
{
    std::ignore = ctx;
    std::ignore = problem;

    const auto dev_name = ctx.GetStream().GetDeviceName();
    const auto cu_count = ctx.GetStream().GetMaxHardwareComputeUnits();
    WinoShaderArgsV2 args;

    // Main convolution parameters
    if(!args.SetConvParams(problem))
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    auto shader_model = ShaderModelFactory::Create(dev_name, args, cu_count, problem);

    auto result = shader_model->ComputeWti(problem.IsFp32());

    return result.wti;
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoRageRxSCommon<Winodata, Winofilter>::GetSolution(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem,
                                                         bool fused,
                                                         bool do_bias,
                                                         miopenActivationMode_t activ_mode)
{
    // Kernel args
    WinoShaderArgsV2 args;
    if(!args.SetConvParams(problem))
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }
    args.SetStrides(problem);
    args.SetActivParams(activ_mode);

    auto flags = WinoShaderFlagsV2::F_NKCHR_STRIDES | WinoShaderFlagsV2::F_TENSOR_OFFSETS |
                 WinoShaderFlagsV2::F_USE_ACTIVATION_MODE |
                 WinoShaderFlagsV2::F_DENORMS_RND_ENABLE |
                 WinoShaderFlagsV2::F_USE_EXTENDED_FLAGS_64;
    if(args.G != 1)
        flags |= WinoShaderFlagsV2::F_GROUPED_CONVOLUTION;
    if(problem.IsDirectionBackwardData())
        flags |= WinoShaderFlagsV2::F_REVERSE_R | WinoShaderFlagsV2::F_REVERSE_S;
    if(do_bias)
        flags |= WinoShaderFlagsV2::F_BIAS;

    const auto devName  = ctx.GetStream().GetDeviceName();
    auto kernel_version = ShaderModelFactory::DetermineKernelVersion(args, problem.IsFp16());
    auto shader_model   = ShaderModelFactory::Create(
        devName, args, ctx.GetStream().GetMaxHardwareComputeUnits(), problem, kernel_version);
    auto perfmodel_result = shader_model->ComputeWti(problem.IsFp32());
    auto nGroups          = perfmodel_result.n_groups;

    args.SetShaderParams(nGroups, flags, 0, 0);

    // Kernel name and file
    std::string kernelVersion;
    if(kernel_version == ShaderModelFactory::KernelVersion::V4_6)
    {
        kernelVersion = "_v4_6_1";
    }
    else // V4_9
    {
        kernelVersion = "_v4_9_0";
    }
    std::string kernelName = "miopenSp3AsmConvRage" + kernelVersion;
    std::string kernelFile = "Conv_Winograd_Rage" + kernelVersion;

    if(devName == "gfx942")
    {
        kernelName += "_gfx9";
    }
    else if(StartsWith(devName, "gfx12"))
    {
        kernelName += "_gfx12";
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    std::string kernelPostfix;

    if(problem.IsFp16())
    {
        kernelPostfix += "_fp16_fp32acc";
    }
    else if(problem.IsFp32())
    {
        kernelPostfix += "_fp32_fp32acc";
    }
    else if(problem.IsBfp16())
    {
        kernelPostfix += "_bf16_fp32acc";
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    if constexpr(Winodata == 2 && Winofilter == 3)
    {
        kernelPostfix += "_f2x3";
    }
    else
    {
        static_assert(Winodata == 2 && Winofilter == 3);
    }

    kernelPostfix += "_stride1";

    kernelName += kernelPostfix;
    kernelFile += kernelPostfix;
    kernelFile += ".s";

    // Kernel info

    KernelInfo kernelInfo;

    /// Kernel doesn't need ROCM_METADATA_VERSION, but AmdgcnAssemble()
    /// uses it to find out required CO version (hack).
    /// \todo Delete when COv2 support is removed.
    KernelBuildParameters options{
        {"ROCM_METADATA_VERSION", 5},
    };
    kernelInfo.comp_options = options.GenerateFor(kbp::GcnAsm{});
    kernelInfo.comp_options += std::string(" -mcumode");

    uint64_t wgSize = 768U; // value for gfx942
    if(StartsWith(devName, "gfx12"))
    {
        wgSize = 384U;
    }

    kernelInfo.l_wk.push_back(wgSize);
    kernelInfo.l_wk.push_back(1);
    kernelInfo.l_wk.push_back(1);

    kernelInfo.g_wk.push_back(wgSize * nGroups * args.G);
    kernelInfo.g_wk.push_back(1);
    kernelInfo.g_wk.push_back(1);

    kernelInfo.kernel_file = kernelFile;
    kernelInfo.kernel_name = kernelName;

    // Solution
    ConvSolution result;
    result.construction_params.push_back(kernelInfo);
    result.invoker_factory =
        miopen::MakeGcnAsmWinoV2InvokerFactory(args, problem.GetDirection(), 0U, fused);
    result.workspace_sz = 0U;

    return result;
}

} // namespace

namespace conv {

template <uint32_t Winodata, uint32_t Winofilter>
bool ConvWinoRageRxS<Winodata, Winofilter>::IsApplicable(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem) const
{
    return ConvWinoRageRxSCommon<Winodata, Winofilter>::IsApplicable(ctx, problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
float ConvWinoRageRxS<Winodata, Winofilter>::GetWti(const ExecutionContext& ctx,
                                                    const ProblemDescription& problem) const
{
    return ConvWinoRageRxSCommon<Winodata, Winofilter>::GetWti(ctx, problem);
}

template <uint32_t Winodata, uint32_t Winofilter>
ConvSolution
ConvWinoRageRxS<Winodata, Winofilter>::GetSolution(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    return ConvWinoRageRxSCommon<Winodata, Winofilter>::GetSolution(ctx, problem);
}

template struct MIOPEN_INTERNALS_EXPORT ConvWinoRageRxS<2, 3>;

} // namespace conv

} // namespace solver

} // namespace miopen
