/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// Regression tests for the TF32 perf-db reuse bug: a CK grouped-conv PerformanceConfig serializes
// only its kernel_id, so a config restored from the perf-db loses use_tf32. GetSolution used to
// trust that (now false) flag and silently build the FP32 instance for a TF32 problem. The fix
// recomputes use_tf32 from the problem at solution-build time, in all six CK grouped solvers
// (2D/3D x fwd/bwd-data/bwd-weights); these tests cover every one of them.

#include <gtest/gtest.h>

#include "group_conv.hpp"

#include <miopen/conv/solvers.hpp>
#include <miopen/tensor.hpp>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

using namespace group_conv;

namespace {

// Coverage signal: DbRestoredConfigStaysTF32 skips where the CK build runs TF32 identically to
// FP32, so a run in which *every* case skipped verified nothing. Track it and warn loudly at exit,
// so a green CI on a no-distinct-TF32 build cannot be mistaken for the regression being exercised.
std::atomic<int> g_reuse_attempted{0};
std::atomic<int> g_reuse_observed{0};
struct ReuseCoverageReporter
{
    ~ReuseCoverageReporter()
    {
        if(g_reuse_attempted.load() > 0 && g_reuse_observed.load() == 0)
            std::cerr
                << "[TF32Reuse] WARNING: all " << g_reuse_attempted.load()
                << " DbRestoredConfigStaysTF32 case(s) skipped (TF32 numerically identical to "
                   "FP32 on this build/device); the perf-db TF32 reuse fix was NOT exercised\n";
    }
};
const ReuseCoverageReporter g_reuse_reporter;

template <unsigned NDIM, Direction DIR>
struct GroupConvTF32Reuse : GroupConvTestFix<NDIM, float, DIR>
{
    using Base = GroupConvTestFix<NDIM, float, DIR>;
    using Base::alpha;
    using Base::beta;
    using Base::compute_type;
    using Base::conv_config;
    using Base::conv_desc;
    using Base::in_dev;
    using Base::input;
    using Base::out_dev;
    using Base::output;
    using Base::test_skipped;
    using Base::wei_dev;
    using Base::weights;
    using Base::wspace;

    // compute_type must be set before the base SetUp() consumes it.
    GroupConvTF32Reuse() { compute_type = "TF32"; }

    // These tests do their own verification, so suppress the base reference-conv check.
    void TearDown() override {}

    // Problem built with the per-direction tensor-descriptor order the solvers expect.
    miopen::conv::ProblemDescription MakeProblem(const miopen::ConvolutionDescriptor& cd)
    {
        if constexpr(DIR == Direction::Forward)
            return {input.desc, weights.desc, output.desc, cd, DIR, 0, alpha, beta};
        else
            return {output.desc, weights.desc, input.desc, cd, DIR, 0, alpha, beta};
    }

    template <class Solver>
    void RunReuse()
    {
        auto& handle = get_handle();
        Solver solv{};
        auto ctx = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        auto problem_tf32 = MakeProblem(conv_desc); // conv_desc has math_type Default for TF32
        problem_tf32.SetupComputeType(ctx);
        if(!problem_tf32.UseTF32())
        {
            test_skipped = true;
            GTEST_SKIP() << "TF32 compute is not enabled on this device";
        }
        if(!solv.IsApplicable(ctx, problem_tf32))
        {
            test_skipped = true;
            GTEST_SKIP() << solv.SolverDbId() << " is not applicable: " << conv_config;
        }

        auto conv_desc_fp32 = conv_desc;
        conv_desc_fp32.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathPedantic);
        auto problem_fp32 = MakeProblem(conv_desc_fp32);
        problem_fp32.SetupComputeType(
            ctx); // keeps workspace/solution sizing consistent (stays FP32)

        if(solv.MayNeedWorkspace())
            wspace.resize(std::max(solv.GetWorkspaceSize(ctx, problem_tf32),
                                   solv.GetWorkspaceSize(ctx, problem_fp32)));

        // Direction-specific invoke tensors and the buffer holding this direction's result. Mirrors
        // GroupConvTestFix::DispatchSolver; its RunSolverImpl cannot be reused here because it
        // always uses the default perf config and never reads the result back for comparison.
        using InvokeParams = std::conditional_t<DIR == Direction::BackwardWeights,
                                                miopen::conv::WrWInvokeParams,
                                                miopen::conv::DataInvokeParams>;
        const auto tensors = [&] {
            if constexpr(DIR == Direction::Forward)
                return miopen::ConvDataTensors{input.desc,
                                               in_dev.get(),
                                               weights.desc,
                                               wei_dev.get(),
                                               output.desc,
                                               out_dev.get()};
            else if constexpr(DIR == Direction::BackwardData)
                return miopen::ConvDataTensors{output.desc,
                                               out_dev.get(),
                                               weights.desc,
                                               wei_dev.get(),
                                               input.desc,
                                               in_dev.get()};
            else
                return miopen::ConvWrwTensors{output.desc,
                                              out_dev.get(),
                                              input.desc,
                                              in_dev.get(),
                                              weights.desc,
                                              wei_dev.get()};
        }();
        auto& result_src = [&]() -> tensor<float>& {
            if constexpr(DIR == Direction::Forward)
                return output;
            else if constexpr(DIR == Direction::BackwardData)
                return input;
            else
                return weights;
        }();
        auto& result_dev = [&]() -> miopen::Allocator::ManageDataPtr& {
            if constexpr(DIR == Direction::Forward)
                return out_dev;
            else if constexpr(DIR == Direction::BackwardData)
                return in_dev;
            else
                return wei_dev;
        }();

        const auto invoke_params =
            InvokeParams{tensors, wspace.ptr(), wspace.size(), false, alpha, beta};

        auto run = [&](const auto& prob, const auto& perf_cfg, tensor<float>& result) {
            auto sol = solv.GetSolution(ctx, prob, perf_cfg);
            ASSERT_TRUE(sol.Succeeded());
            ASSERT_TRUE(sol.invoker_factory); // guard the deref below
            const auto invoker =
                handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
            (invoker)(handle, invoke_params);
            handle.Finish();
            handle.ReadToVec(result_dev, result.data);
        };

        // Heuristic config for the TF32 problem, round-tripped through perf-db serialization: keeps
        // kernel_id, clears use_tf32 -- exactly the state of a config after a perf-db reload.
        auto cfg = solv.GetDefaultPerformanceConfig(ctx, problem_tf32);
        std::ostringstream oss;
        cfg.Serialize(oss);
        std::decay_t<decltype(cfg)> restored;
        ASSERT_TRUE(restored.Deserialize(oss.str()));
        ASSERT_EQ(restored.kernel_id, cfg.kernel_id);
        ASSERT_FALSE(restored.UseTF32())
            << "test precondition: a reloaded perf-db config must have dropped use_tf32";

        tensor<float> out_tf32 = result_src;
        run(problem_tf32, restored, out_tf32);
        tensor<float> out_fp32 = result_src;
        run(problem_fp32, restored, out_fp32);
        tensor<float> out_ref = result_src; // pristine TF32 config, no db round-trip
        run(problem_tf32, cfg, out_ref);

        ASSERT_EQ(miopen::range_distance(out_tf32), miopen::range_distance(out_fp32));

        // Some CK builds carry no numerically distinct TF32 kernel for this instance on this
        // device; there TF32 == FP32 and the reuse property is unobservable, so skip rather than
        // false-fail. Count attempts/observations so an all-skipped run is reported at exit.
        g_reuse_attempted.fetch_add(1);
        if(miopen::rms_range(out_fp32, out_ref) <= 1e-5)
        {
            test_skipped = true;
            GTEST_SKIP() << "CK runs TF32 identically to FP32 on this device; cannot observe reuse";
        }
        g_reuse_observed.fetch_add(1);

        // The fix: a db-restored config must still run TF32 -- matching the pristine TF32 result
        // and diverging from FP32 -- not silently fall back to FP32.
        EXPECT_GT(miopen::rms_range(out_fp32, out_tf32), 1e-5)
            << "perf-db-restored TF32 config produced an FP32-identical result";
        EXPECT_LE(miopen::rms_range(out_ref, out_tf32), 1e-6)
            << "perf-db-restored TF32 result diverged from the pristine TF32 result";
    }

    // TF32 and strict-FP32 versions of a problem must serialize to different perf-db keys (the
    // _cxTF32 marker) so their tuned records never collide. The marker needs SetupComputeType on a
    // TF32-capable device, so this is a GPU test that skips elsewhere.
    void RunKeyDistinct()
    {
        auto& handle = get_handle();
        auto ctx     = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        auto problem_tf32 = MakeProblem(conv_desc);
        problem_tf32.SetupComputeType(ctx);
        if(!problem_tf32.UseTF32())
        {
            test_skipped = true;
            GTEST_SKIP() << "TF32 compute is not enabled on this device";
        }

        auto conv_desc_fp32 = conv_desc;
        conv_desc_fp32.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathPedantic);
        auto problem_fp32 = MakeProblem(conv_desc_fp32);
        problem_fp32.SetupComputeType(ctx);

        std::ostringstream key_tf32;
        std::ostringstream key_fp32;
        problem_tf32.Serialize(key_tf32);
        problem_fp32.Serialize(key_fp32);

        EXPECT_NE(key_tf32.str().find("cxTF32"), std::string::npos)
            << "TF32 problem key is missing the compute marker: " << key_tf32.str();
        EXPECT_EQ(key_fp32.str().find("cxTF32"), std::string::npos)
            << "strict-FP32 key unexpectedly carries the TF32 marker: " << key_fp32.str();
        EXPECT_NE(key_tf32.str(), key_fp32.str())
            << "TF32 and FP32 problems serialize to the same perf-db key; their records collide";
    }
};

template <unsigned NDIM>
std::vector<GroupConvTestConfig<NDIM>> Tf32ReuseConfigs()
{
    if constexpr(NDIM == 2u)
        return {{8, 256, 384, 384, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}}};
    else
        return {{8, 64, 64, 64, {4, 16, 16}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};
}

template <unsigned NDIM>
miopenTensorLayout_t Tf32ReuseLayout()
{
    return NDIM == 2u ? miopenTensorNHWC : miopenTensorNDHWC;
}

} // namespace

// One suite per CK grouped solver the fix touches (2D/3D x fwd/bwd-data/bwd-weights). Suite
// names are spelled out rather than macro-pasted so the MIOpen gtest-naming linter can parse
// them.

struct GPU_GroupConv2D_Forward_TF32Reuse_FP32 : GroupConvTF32Reuse<2u, Direction::Forward>
{
};
TEST_P(GPU_GroupConv2D_Forward_TF32Reuse_FP32, DbRestoredConfigStaysTF32)
{
    RunReuse<miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops>();
}
TEST_P(GPU_GroupConv2D_Forward_TF32Reuse_FP32, TF32AndFp32GetDistinctDbKeys) { RunKeyDistinct(); }
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GroupConv2D_Forward_TF32Reuse_FP32,
                         testing::Combine(testing::ValuesIn(Tf32ReuseConfigs<2u>()),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f),
                                          testing::Values(Tf32ReuseLayout<2u>())));

struct GPU_GroupConv2D_BackwardData_TF32Reuse_FP32 : GroupConvTF32Reuse<2u, Direction::BackwardData>
{
};
TEST_P(GPU_GroupConv2D_BackwardData_TF32Reuse_FP32, DbRestoredConfigStaysTF32)
{
    RunReuse<miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops>();
}
TEST_P(GPU_GroupConv2D_BackwardData_TF32Reuse_FP32, TF32AndFp32GetDistinctDbKeys)
{
    RunKeyDistinct();
}
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GroupConv2D_BackwardData_TF32Reuse_FP32,
                         testing::Combine(testing::ValuesIn(Tf32ReuseConfigs<2u>()),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f),
                                          testing::Values(Tf32ReuseLayout<2u>())));

struct GPU_GroupConv2D_BackwardWeights_TF32Reuse_FP32
    : GroupConvTF32Reuse<2u, Direction::BackwardWeights>
{
};
TEST_P(GPU_GroupConv2D_BackwardWeights_TF32Reuse_FP32, DbRestoredConfigStaysTF32)
{
    RunReuse<miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops>();
}
TEST_P(GPU_GroupConv2D_BackwardWeights_TF32Reuse_FP32, TF32AndFp32GetDistinctDbKeys)
{
    RunKeyDistinct();
}
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GroupConv2D_BackwardWeights_TF32Reuse_FP32,
                         testing::Combine(testing::ValuesIn(Tf32ReuseConfigs<2u>()),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f),
                                          testing::Values(Tf32ReuseLayout<2u>())));

struct GPU_GroupConv3D_Forward_TF32Reuse_FP32 : GroupConvTF32Reuse<3u, Direction::Forward>
{
};
TEST_P(GPU_GroupConv3D_Forward_TF32Reuse_FP32, DbRestoredConfigStaysTF32)
{
    RunReuse<miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops>();
}
TEST_P(GPU_GroupConv3D_Forward_TF32Reuse_FP32, TF32AndFp32GetDistinctDbKeys) { RunKeyDistinct(); }
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GroupConv3D_Forward_TF32Reuse_FP32,
                         testing::Combine(testing::ValuesIn(Tf32ReuseConfigs<3u>()),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f),
                                          testing::Values(Tf32ReuseLayout<3u>())));

struct GPU_GroupConv3D_BackwardData_TF32Reuse_FP32 : GroupConvTF32Reuse<3u, Direction::BackwardData>
{
};
TEST_P(GPU_GroupConv3D_BackwardData_TF32Reuse_FP32, DbRestoredConfigStaysTF32)
{
    RunReuse<miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops>();
}
TEST_P(GPU_GroupConv3D_BackwardData_TF32Reuse_FP32, TF32AndFp32GetDistinctDbKeys)
{
    RunKeyDistinct();
}
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GroupConv3D_BackwardData_TF32Reuse_FP32,
                         testing::Combine(testing::ValuesIn(Tf32ReuseConfigs<3u>()),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f),
                                          testing::Values(Tf32ReuseLayout<3u>())));

struct GPU_GroupConv3D_BackwardWeights_TF32Reuse_FP32
    : GroupConvTF32Reuse<3u, Direction::BackwardWeights>
{
};
TEST_P(GPU_GroupConv3D_BackwardWeights_TF32Reuse_FP32, DbRestoredConfigStaysTF32)
{
    RunReuse<miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops>();
}
TEST_P(GPU_GroupConv3D_BackwardWeights_TF32Reuse_FP32, TF32AndFp32GetDistinctDbKeys)
{
    RunKeyDistinct();
}
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GroupConv3D_BackwardWeights_TF32Reuse_FP32,
                         testing::Combine(testing::ValuesIn(Tf32ReuseConfigs<3u>()),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f),
                                          testing::Values(Tf32ReuseLayout<3u>())));
