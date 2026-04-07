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
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/ck_grouped_conv_lib_loader.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_DEFAULT_KERNELS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
void PerformanceConfigHipImplicitGemmGroupBwdXdlops::InitHeuristicKernelIDs()
{
    for(int i = 0; i < valid_kernels.size(); i++)
    {
        if(valid_kernels[i].find("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1") !=
           std::string::npos)
        {
            heuristic_indexes.push_back(i);
            heuristic_kernels[i] = GetKernelAsTokens(valid_kernels[i]);
        }
    }
}

bool PerformanceConfigHipImplicitGemmGroupBwdXdlops::ModelApplyToken(
    int idx, std::string value, const std::string& arch, const ProblemDescription& /*problem*/)
{
    if(arch == "gfx90a")
    {
        if(idx == 13)
            idx += 1; // skip
    }
    if(arch == "gfx942" || arch == "gfx950")
    {
        if(idx < 3)
            idx += 0;
        else if(idx <= 4)
            idx += 2;
        else if(idx <= 8)
            idx += 4;
        else
            return true;
    }

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
GetFeatures(const ProblemDescription& problem, std::size_t /*num_cu*/, const std::string& arch)
{
    if(arch == "gfx90a")
    {
        std::size_t n = 18;
        std::vector<float> features(n * n, 0.0f);
        features[0]           = 0.0;
        features[n + 1]       = problem.GetOutChannels();
        features[2 * n + 2]   = problem.GetOutHeight();
        features[3 * n + 3]   = problem.GetOutWidth();
        features[4 * n + 4]   = problem.GetInChannels();
        features[5 * n + 5]   = problem.GetInHeight();
        features[6 * n + 6]   = problem.GetInWidth();
        features[7 * n + 7]   = problem.GetWeightsHeight();
        features[8 * n + 8]   = problem.GetWeightsWidth();
        features[9 * n + 9]   = problem.GetPadH();
        features[10 * n + 10] = problem.GetPadW();
        features[11 * n + 11] = problem.GetKernelStrideH();
        features[12 * n + 12] = problem.GetKernelStrideW();
        features[13 * n + 13] = problem.GetDilationH();
        features[14 * n + 14] = problem.GetDilationW();
        features[15 * n + 15] = problem.GetBatchSize();
        features[16 * n + 16] = problem.GetInDataType() == miopenFloat ? 2.0 : 1.0;
        features[17 * n + 17] = problem.GetGroupCount();
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

bool PerformanceConfigHipImplicitGemmGroupBwdXdlops::RunParameterPredictionModel(
    const ExecutionContext& ctx, const ProblemDescription& problem)
{
    const auto& loader =
        miopen::solver::CKGroupedConvLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return false;

    auto data_type = problem.GetInDataType();
    use_tf32       = (data_type == miopenFloat) && problem.UseTF32();

    valid_kernels =
        loader.FillValidKernelsWithTf32Fallback(CKConvDirection::Bwd, problem, data_type, use_tf32);
    if(valid_kernels.empty())
        return false;

    InitHeuristicKernelIDs();
    const auto arch    = ctx.GetStream().GetDeviceName();
    std::string solver = "ConvHipIgemmGroupBwdXdlops";
    if(arch == "gfx90a")
        solver = "ConvHipIgemmGroupXdlops";
    std::vector<float> features = GetFeatures(problem, ctx.GetStream().GetMaxComputeUnits(), arch);
    if(ai::tuning::ModelSetParams(
           arch, solver, problem.GetDirection(), features, true, [&](int idx, std::string value) {
               return this->ModelApplyToken(idx, value, arch, problem);
           }))
    {
        index     = heuristic_indexes[0];
        kernel_id = valid_kernels[index] + "+1";
        MIOPEN_LOG_I("Params set by AI: " << ToString());
        return true;
    }
    return false;
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING

bool PerformanceConfigHipImplicitGemmGroupBwdXdlops::IsModelApplicable(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(ctx.GetStream().GetDeviceName() != "gfx90a" && ctx.GetStream().GetDeviceName() != "gfx942" &&
       ctx.GetStream().GetDeviceName() != "gfx950")
        return false;
    if(problem.GetInDataType() != miopenFloat && problem.GetInDataType() != miopenHalf &&
       problem.GetInDataType() != miopenBFloat16)
        return false;
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR))
        return false;
    return true;
}

// clang-format off
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::tuple<std::string, int>> ranked_gemm_grp_bwd = {
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 64, 16, 16, Filter1x1Stride1Pad0, 32, 32, 1, 1, 16, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 16, 4, 4, Filter1x1Stride1Pad0, 32, 32, 1, 1, 4, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 16, 4, 4, Filter1x1Stride1Pad0, 32, 32, 1, 1, 4, 2, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 16, 4, 4, Filter1x1Stride1Pad0, 32, 32, 1, 1, 4, 1, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 128, 32, 8, 8, Filter1x1Stride1Pad0, 32, 32, 1, 2, 1, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 64, 16, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 4, 1, 1, 4, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 64, 64, 32, 8, 8, Filter1x1Stride1Pad0, 32, 32, 2, 2, 1, 1, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 64, 16, 16, Default, 32, 32, 1, 1, 16, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 64, 16, 16, Default, 16, 16, 1, 1, 16, 1, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<128, 128, 32, 32, 8, 8, Default, 32, 32, 2, 1, 8, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 32, 8, 8, Default, 16, 16, 1, 1, 8, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 32, 8, 8, Default, 32, 32, 1, 1, 8, 2, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 32, 8, 8, Default, 16, 16, 1, 1, 8, 1, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 16, 4, 4, Default, 32, 32, 1, 1, 4, 2, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 64, 64, 32, 8, 8, Default, 32, 32, 2, 2, 1, 1, 1, 1>", 1)
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::tuple<std::string, int>> ranked_gemm_grp_bwd_navi = {
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<128, 128, 64, 64, Filter1x1Stride1Pad0, 8, 8, 2>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Default, 16, 16, 1, 4, 8, 1, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<256, 128, 64, 64, Filter1x1Stride1Pad0, 8, 8, 2>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 1, 4, 8, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<64, 64, 32, 64, 8, 8, Default, 16, 16, 4, 1, 8, 4, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<128, 64, 64, 32, Default, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<64, 64, 64, 32, 8, 8, Default, 16, 16, 4, 2, 1, 1, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<256, 128, 64, 64, Default, 8, 8, 2>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<64, 64, 64, 64, Default, 8, 8, 8>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<128, 64, 64, 32, Filter1x1Stride1Pad0, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<64, 64, 64, 32, 8, 8, Default, 16, 16, 4, 2, 4, 4, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<64, 32, 64, 64, Default, 8, 8, 8>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Default, 16, 16, 1, 4, 1, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<128, 128, 64, 64, Default, 8, 8, 2>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<128, 128, 128, 32, 8, 8, Default, 16, 16, 8, 2, 8, 4, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<128, 128, 64, 64, 8, 8, Default, 16, 16, 8, 1, 8, 4, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<256, 128, 256, 64, Default, 8, 8, 8>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Default, 16, 16, 1, 4, 8, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<64, 64, 64, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 4, 2, 1, 1, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 1, 4, 8, 1, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 16, 64, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 1, 4, 1, 8, 1, 1>", 1),
std::make_tuple("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<64, 64, 32, 32, 8, 8, Default, 16, 16, 4, 1, 8, 1, 1, 1>", 1)
};
// clang-format on

void PerformanceConfigHipImplicitGemmGroupBwdXdlops::DefaultKernelFromList(
    const ExecutionContext& ctx)
{
    const auto dev_name = ctx.GetStream().GetDeviceName();
    const bool is_gfx11 = StartsWith(dev_name, "gfx11");
    const bool is_gfx12 = StartsWith(dev_name, "gfx12");

    auto* ranked_p = &ranked_gemm_grp_bwd;
    if(is_gfx11 || is_gfx12)
        ranked_p = &ranked_gemm_grp_bwd_navi;

    const auto ranked_1st_applicable = *ranked_p;

    for(const auto& kernel : ranked_1st_applicable)
    {
        const auto& kernel_str = std::get<0>(kernel);
        const auto& it         = std::find(valid_kernels.begin(), valid_kernels.end(), kernel_str);
        if(it != valid_kernels.end())
        {
            index     = it - valid_kernels.begin();
            split_k   = 1;
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            return;
        }
    }
}

void PerformanceConfigHipImplicitGemmGroupBwdXdlops::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    split_k   = 1;
    index     = 0;
    kernel_id = "";

    const auto& loader =
        miopen::solver::CKGroupedConvLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return;

    const bool is_deterministic = problem.GetConv().attribute.deterministic;

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(IsModelApplicable(ctx, problem))
    {
        if(RunParameterPredictionModel(ctx, problem))
        {
            // Enforce split_k == 1 for deterministic mode
            if(is_deterministic && split_k != 1)
            {
                MIOPEN_LOG_I("Deterministic mode: Overriding AI-predicted split_k="
                             << split_k << " to split_k=1");
                split_k = 1;
                if(!valid_kernels.empty())
                    kernel_id = valid_kernels[index] + "+1";
            }
            return;
        }
    }
#endif

    auto data_type = problem.GetInDataType();
    use_tf32       = (data_type == miopenFloat) && problem.UseTF32();

    valid_kernels =
        loader.FillValidKernelsWithTf32Fallback(CKConvDirection::Bwd, problem, data_type, use_tf32);

    if(!valid_kernels.empty())
    {
        index     = 0;
        split_k   = 1;
        kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
    }

    if(!env::disabled(MIOPEN_DEBUG_CK_DEFAULT_KERNELS))
        DefaultKernelFromList(ctx);

    // Invariant: split_k must always be 1 in deterministic mode
    assert(!is_deterministic || split_k == 1);
}

bool PerformanceConfigHipImplicitGemmGroupBwdXdlops::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        const auto& loader = miopen::solver::CKGroupedConvLibLoader::Get(GetCurrentDeviceName());
        if(!loader.IsLoaded())
            return false;

        auto data_type = problem.GetInDataType();
        use_tf32       = (data_type == miopenFloat) && problem.UseTF32();

        valid_kernels = loader.FillValidKernelsWithTf32Fallback(
            CKConvDirection::Bwd, problem, data_type, use_tf32);

        if(valid_kernels.empty())
            return false;

        assert(!valid_kernels.empty());
        return true;
    }

    const bool is_deterministic = problem.GetConv().attribute.deterministic;

    // Deterministic mode: only iterate over kernels (index), split_k is always 1
    if(is_deterministic)
    {
        if(!NextLinear(0, valid_kernels.size() - 1, index))
        {
            return false; // All kernels exhausted
        }
        split_k   = 1;
        kernel_id = valid_kernels[index] + "+1";
        return true;
    }

    // General (non-deterministic) mode: iterate over both split_k and kernels
    do
    {
        bool flag = NextTwoPower<1, 128>(split_k);

        if(!flag)
        {
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            break;
        }

        if(!NextLinear(0, valid_kernels.size() - 1, index))
        {
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            break;
        }
        // All split_k and index values were iterated
        return false;
    } while(false);
    return true;
}

bool PerformanceConfigHipImplicitGemmGroupBwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmGroupBwdXdlops::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
    if(!IsDeterministicSplitKValid(kernel_id, problem.GetConv().attribute.deterministic))
        return false;

    const auto& loader = miopen::solver::CKGroupedConvLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;

    auto data_type = problem.GetInDataType();
    return loader.IsArgsSupported(CKConvDirection::Bwd, problem, kernel_id, data_type, use_tf32);
}

bool PerformanceConfigHipImplicitGemmGroupBwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmGroupBwdXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmGroupBwdXdlops
ConvHipImplicitGemmGroupBwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmGroupBwdXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemmGroupBwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmGroupBwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t
ConvHipImplicitGemmGroupBwdXdlops::GetCKMaxWorkspaceSize(const ProblemDescription& problem) const
{
    const auto& loader = CKGroupedConvLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return 0;

    auto data_type = problem.GetInDataType();
    bool use_tf32  = (data_type == miopenFloat) && problem.UseTF32();
    return loader.GetWorkspaceSize(CKConvDirection::Bwd, problem, data_type, use_tf32);
}

size_t ConvHipImplicitGemmGroupBwdXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                           const ProblemDescription& problem) const
{
    auto ck_ws_size = GetCKMaxWorkspaceSize(problem);
    return GetWorkspaceSizeLayoutTransformConv(problem, ck_ws_size);
}

PerformanceConfigHipImplicitGemmGroupBwdXdlops
ConvHipImplicitGemmGroupBwdXdlops::Search(const ExecutionContext& ctx,
                                          const ProblemDescription& problem,
                                          const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmGroupBwdXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
    if(env::enabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS))
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;
    // needed because layout transpose kernel does not support non-packed tensors
    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
        return false;

    const auto& loader = CKGroupedConvLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return false;

    return loader.IsApplicable(
        CKConvDirection::Bwd, problem, problem.GetInDataType(), problem.UseTF32());
}

ConvSolution ConvHipImplicitGemmGroupBwdXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmGroupBwdXdlops& config) const
{
    const auto& loader = CKGroupedConvLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return {};

    return loader.GetSolution(
        CKConvDirection::Bwd, ctx, problem, config.kernel_id, config.UseTF32());
}

} // namespace conv
} // namespace solver
} // namespace miopen
