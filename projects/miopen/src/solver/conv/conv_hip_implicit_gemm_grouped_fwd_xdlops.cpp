/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>
#include <cstdint>

#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_DEFAULT_KERNELS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

namespace {

// ============================================================================
// Solver Configuration for FWD 2D Grouped Convolution
// ============================================================================

// clang-format off
constexpr SolverHeuristicConfig k2DFwdSolverConfig = {
    /* solver_name                 */ "ConvHipImplicitGemmGroupFwdXdlops",
    /* solver_name_ktn             */ "ConvHipIgemmGroupFwdXdlops",
    /* spatial_dims                */ 2,
    /* uses_split_k                */ false,
    /* split_k_min                 */ 0,
    /* split_k_max                 */ 0,
    /* supports_split_k_autodeduce */ false,
    /* supports_ktn                */ true,
};
// clang-format on

} // namespace

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
void PerformanceConfigHipImplicitGemmGroupFwdXdlops::InitHeuristicKernelIDsKTN(
    const std::string& type)
{
    for(int i = 0; i < valid_kernels.size(); i++)
    {
        if(valid_kernels[i].find(type) != std::string::npos)
        {
            heuristic_indexes.push_back(i);
            heuristic_kernels[i] = GetKernelAsTokens2D(valid_kernels[i]);
        }
    }
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::ModelApplyTokenKTN(int idx,
                                                                        std::string value,
                                                                        const std::string& arch)
{
    if(arch == "gfx90a")
    {
        if(idx >= 5)
        {
            idx += 2; // skip MPerXDL and NPerXDL as they are constant
        }
    }
    if(idx == 0 && (arch == "gfx942" || arch == "gfx950"))
    {
        InitHeuristicKernelIDsKTN(value);
        if(!heuristic_indexes.empty())
            return true;
        return false;
    }
    if(idx >= 1 && (arch == "gfx942" || arch == "gfx950"))
        idx--;
    auto eraseBegin = std::remove_if(
        heuristic_indexes.begin(), heuristic_indexes.end(), [&](int heuristic_index) {
            return heuristic_kernels[heuristic_index][idx] != value;
        });

    if(eraseBegin != heuristic_indexes.begin())
    {
        heuristic_indexes.erase(eraseBegin, heuristic_indexes.end());
        return true;
    }
    return false;
}

static std::vector<float>
GetFeaturesKTN(const ProblemDescription& problem, std::size_t num_cu, const std::string& arch)
{
    if(arch == "gfx90a")
    {
        std::size_t n = 18;
        std::vector<float> features(n, 0.0f);
        features[0]  = problem.GetInDataType() == miopenFloat ? 2 : 1;
        features[1]  = problem.GetInChannels();
        features[2]  = problem.GetInHeight();
        features[3]  = problem.GetInWidth();
        features[4]  = problem.GetOutChannels();
        features[5]  = problem.GetOutHeight();
        features[6]  = problem.GetOutWidth();
        features[7]  = problem.GetWeightsHeight();
        features[8]  = problem.GetWeightsWidth();
        features[9]  = problem.GetPadH();
        features[10] = problem.GetPadW();
        features[11] = problem.GetKernelStrideH();
        features[12] = problem.GetKernelStrideW();
        features[13] = problem.GetDilationH();
        features[14] = problem.GetDilationW();
        features[15] = problem.GetBatchSize();
        features[16] = problem.GetGroupCount();
        features[17] = num_cu;
        return features;
    }

    const bool isFwd = problem.GetDirection() == miopen::conv::Direction::Forward;
    float precision  = 2.0; // miopenHalf
    if(problem.GetInDataType() == miopenFloat)
        precision = 3.0;
    else if(problem.GetInDataType() == miopenBFloat16)
        precision = 1.0;

    std::size_t n = 17;
    std::vector<float> features(n * n, 0.0f);
    features[0]           = isFwd ? problem.GetInChannels() : problem.GetOutChannels();
    features[n + 1]       = isFwd ? problem.GetInHeight() : problem.GetOutHeight();
    features[2 * n + 2]   = isFwd ? problem.GetInWidth() : problem.GetOutWidth();
    features[3 * n + 3]   = isFwd ? problem.GetOutChannels() : problem.GetInChannels();
    features[4 * n + 4]   = isFwd ? problem.GetOutHeight() : problem.GetInHeight();
    features[5 * n + 5]   = isFwd ? problem.GetOutWidth() : problem.GetInWidth();
    features[6 * n + 6]   = problem.GetWeightsHeight();
    features[7 * n + 7]   = problem.GetWeightsWidth();
    features[8 * n + 8]   = problem.GetPadH();
    features[9 * n + 9]   = problem.GetPadW();
    features[10 * n + 10] = problem.GetKernelStrideH();
    features[11 * n + 11] = problem.GetKernelStrideW();
    features[12 * n + 12] = problem.GetDilationH();
    features[13 * n + 13] = problem.GetDilationW();
    features[14 * n + 14] = problem.GetBatchSize();
    features[15 * n + 15] = precision;
    features[16 * n + 16] = problem.GetGroupCount();
    return features;
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::RunParameterPredictionModelKTN(
    const ExecutionContext& ctx, const ProblemDescription& problem)
{
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return false;

    auto data_type = problem.GetInDataType();
    bool try_tf32  = (data_type == miopenFloat) && problem.UseTF32();

    valid_kernels = loader.FillValidKernelsWithTf32Fallback(
        CKSolverType::GrpConvFwd, problem, data_type, try_tf32);

    const auto arch = ctx.GetStream().GetDeviceName();
    if(arch == "gfx90a")
        InitHeuristicKernelIDsKTN("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle");
    const std::string solver = k2DFwdSolverConfig.GetSolverNameForArch(arch);
    std::vector<float> features =
        GetFeaturesKTN(problem, ctx.GetStream().GetMaxComputeUnits(), arch);
    bool transform = (arch == "gfx90a") ? false : true;
    if(ai::tuning::ModelSetParams(arch,
                                  solver,
                                  problem.GetDirection(),
                                  features,
                                  transform,
                                  [&](int idx, const std::string& value) {
                                      return this->ModelApplyTokenKTN(idx, value, arch);
                                  }))
    {
        index     = heuristic_indexes[0];
        kernel_id = valid_kernels[index];
        MIOPEN_LOG_I("Params set by KTN: " << ToString());
        return true;
    }
    return false;
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::IsModelApplicable(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(ctx.GetStream().GetDeviceName() != "gfx90a" && ctx.GetStream().GetDeviceName() != "gfx942" &&
       ctx.GetStream().GetDeviceName() != "gfx950")
        return false;
    if(problem.GetInDataType() != miopenFloat && problem.GetInDataType() != miopenHalf &&
       problem.GetInDataType() != miopenBFloat16)
        return false;
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR))
        return false;
    return true;
}

// clang-format off
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::string> ranked_gemm_grp_fwd = {
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 32, Filter3x3, 16, 16, 4, 1, 4, 1, 1, 1, 1, 16>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 32, Default, 16, 16, 4, 1, 4, 1, 1, 1, 1, 8>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Filter3x3, 16, 16, 4, 1, 4, 1, 1, 1, 1, 16>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, 1, 1, 1, 8>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 128, 64, Filter1x1Pad0, 32, 32, 1, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 32, Filter1x1Stride1Pad0, 32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 64, 64, Default, 32, 32, 1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 128, 32, Filter1x1Stride1Pad0, 32, 32, 4, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Filter1x1Stride1Pad0, 16, 16, 1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Pad0, 16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v2>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Pad0, 32, 32, 2, 1, 4, 4, 4, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Filter1x1Pad0, 32, 32, 2, 1, 4, 4, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 16, Filter1x1Pad0, 32, 32, 2, 2, 1, 1, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 128, 64, OddC, 32, 32, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_DirectLoad<256, 64, 64, 64, Default, 16, 16, 2, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, 2, 4, 4, 4, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, 32, 2, 1, 8, 8, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 2, 2, 2, 1, 2, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, 32, 2, 2, 1, 1, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_DirectLoad<128, 16, 32, 64, Default, 16, 16, 1, 1, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v4>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 32, 16, 64, Default, 16, 16, 1, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 16, Default, 32, 32, 2, 1, 4, 4, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 4, 4, 4, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 32, Default, 32, 32, 2, 2, 1, 1, 8, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, OddC, 32, 32, 2, 2, 1, 1, 4, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<64, 64, 64, 16, Default, 32, 32, 2, 2, 1, 1, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 2, 1, 2, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Default, 32, 32, 2, 2, 1, 1, 1, 1, 1, 1>"
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::string> ranked_gemm_grp_fwd_navi = {
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<64, 64, 16, 16, Default, 16, 16, 4, 1, 4, 1, 1, 1, 1, 16>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<128, 64, 64, 64, Filter1x1Stride1Pad0, 16, 16, 2, 2, 8, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<128, 64, 128, 32, Filter1x1Pad0, 16, 16, 4, 2, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<256, 128, 128, 32, Filter1x1Stride1Pad0, 16, 16, 4, 2, 1, 1, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<256, 64, 64, 32, Filter1x1Pad0, 16, 16, 1, 2, 1, 2, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<128, 128, 64, 32, Default, 32, 32, 2, 2, 8, 8, 8, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<128, 64, 64, 64, Default, 16, 16, 2, 2, 8, 8, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<256, 64, 64, 32, Default, 16, 16, 2, 2, 4, 4, 4, 1, 1, 1>",
"DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<64, 64, 64, 32, Default, 16, 16, 4, 2, 1, 1, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
"DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<32, 16, 16, 64, OddC, 16, 16, 16, 1, 1> AEnableLds: 1, BEnableLds: 1, ABlockTransferSrcScalarPerVector: 16, BBlockTransferSrcScalarPerVector: 16"
};
// clang-format on

void PerformanceConfigHipImplicitGemmGroupFwdXdlops::InitValidKernels(
    const ProblemDescription& problem)
{
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return;

    auto data_type = problem.GetInDataType();
    use_tf32       = (data_type == miopenFloat) && problem.UseTF32();

    valid_kernels = loader.FillValidKernelsWithTf32Fallback(
        CKSolverType::GrpConvFwd, problem, data_type, use_tf32);

    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[index];
    }
}

void PerformanceConfigHipImplicitGemmGroupFwdXdlops::DefaultKernelFromList(
    const ExecutionContext& ctx)
{
    const auto dev_name = ctx.GetStream().GetDeviceName();
    const bool is_gfx11 = StartsWith(dev_name, "gfx11");
    const bool is_gfx12 = StartsWith(dev_name, "gfx12");

    auto* ranked_p = &ranked_gemm_grp_fwd;
    if(is_gfx11 || is_gfx12)
        ranked_p = &ranked_gemm_grp_fwd_navi;

    const auto ranked_1st_applicable = *ranked_p;

    for(const auto& kernel_str : ranked_1st_applicable)
    {
        auto it = std::find(valid_kernels.begin(), valid_kernels.end(), kernel_str);
        if(it != valid_kernels.end())
        {
            index     = it - valid_kernels.begin();
            kernel_id = valid_kernels[index];
            return;
        }
    }
}

void PerformanceConfigHipImplicitGemmGroupFwdXdlops::HeuristicInit(
    const ExecutionContext& ctx, const ProblemDescription& problem)
{
    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);
    state.Reset(k2DFwdSolverConfig.uses_split_k);

    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return;

        // AI heuristics (if enabled)
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(&ctx != &GetDummyCtx() &&
       !env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR))
    {
        bool mode_use_tf32 = (problem.GetInDataType() == miopenFloat) && problem.UseTF32();

        auto fill_valid_kernels = [&loader](const ProblemDescription& p, bool try_tf32) {
            return loader.FillValidKernels(
                CKSolverType::GrpConvFwd, p, p.GetInDataType(), try_tf32);
        };

        auto ktn_runner = [this](const ExecutionContext& c, const ProblemDescription& p) {
            return RunKTNGeneric(*this, c, p);
        };

        auto ck_val_creator = MakeCKValidatorCreator(
            loader, CKSolverType::GrpConvFwd, problem.GetInDataType(), mode_use_tf32);

        if(RunAIHeuristics(k2DFwdSolverConfig,
                           state,
                           ctx,
                           problem,
                           false,
                           fill_valid_kernels,
                           ktn_runner,
                           ck_val_creator,
                           mode_use_tf32))
        {
            return;
        }
    }
#endif

    // Fallback to default initialization
    InitValidKernels(problem);
    if(!valid_kernels.empty())
    {
        if(!env::disabled(MIOPEN_DEBUG_CK_DEFAULT_KERNELS))
            DefaultKernelFromList(ctx);
        state.SetResult(index, split_k, k2DFwdSolverConfig.uses_split_k);
    }
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        const auto& loader = miopen::solver::CkImplLibLoader::Get(GetCurrentDeviceName());
        if(!loader.IsLoaded())
            return false;

        auto data_type = problem.GetInDataType();
        use_tf32       = (data_type == miopenFloat) && problem.UseTF32();

        valid_kernels = loader.FillValidKernelsWithTf32Fallback(
            CKSolverType::GrpConvFwd, problem, data_type, use_tf32);

        if(valid_kernels.empty())
            return false;

        assert(!valid_kernels.empty());
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
        return false;
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
    const auto& loader = miopen::solver::CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;

    auto data_type = problem.GetInDataType();
    return loader.IsArgsSupported(
        CKSolverType::GrpConvFwd, problem, kernel_id, data_type, use_tf32);
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmGroupFwdXdlops& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmGroupFwdXdlops
ConvHipImplicitGemmGroupFwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmGroupFwdXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemmGroupFwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmGroupFwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t ConvHipImplicitGemmGroupFwdXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                           const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemmGroupFwdXdlops
ConvHipImplicitGemmGroupFwdXdlops::Search(const ExecutionContext& ctx,
                                          const ProblemDescription& problem,
                                          const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmGroupFwdXdlops::IsApplicable(const ExecutionContext& ctx,
                                                     const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;
    // needed because layout transpose kernel does not support non-packed tensors
    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
        return false;

    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return false;

    auto data_type = problem.GetInDataType();
    bool try_tf32  = (data_type == miopenFloat) && problem.UseTF32();

    if(try_tf32 && loader.IsApplicable(CKSolverType::GrpConvFwd, problem, data_type, true))
        return true;

    return loader.IsApplicable(CKSolverType::GrpConvFwd, problem, data_type, false);
}

ConvSolution ConvHipImplicitGemmGroupFwdXdlops::GetSolution(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmGroupFwdXdlops& config) const
{
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(
        CKSolverType::GrpConvFwd, ctx, problem, config.kernel_id, config.UseTF32());
}

} // namespace conv
} // namespace solver
} // namespace miopen
