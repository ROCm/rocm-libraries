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
// only its kernel_id, so a config restored from the perf-db loses use_tf32. GetSolution (and the
// IsValid gate ahead of it) used to trust that (now false) flag and silently select the FP32
// instance for a TF32 problem. The fix recomputes use_tf32 from the problem, in all six CK grouped
// solvers (2D/3D x fwd/bwd-data/bwd-weights); these tests cover every one of them.

#include <gtest/gtest.h>

#include "group_conv.hpp"

#include <miopen/conv/solvers.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>
#include <miopen/tensor.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace group_conv;

namespace {

// Two results count as "the same kernel" below kMatchTol and "distinct precisions" above
// kDistinctTol. TF32 vs FP32 of the same convolution differ well above kDistinctTol when the build
// has a distinct TF32 kernel, and the same precision re-run repeats below kMatchTol.
constexpr double kDistinctTol = 1e-5;
constexpr double kMatchTol    = 1e-6;

// Coverage signal: DbRestoredConfigStaysTF32 skips where the CK build runs TF32 identically to
// FP32, so a run in which *every* case skipped verified nothing. Track it and warn loudly at exit,
// so a green CI on a no-distinct-TF32 build cannot be mistaken for the regression being exercised.
// gtest runs these single-threaded, so plain ints suffice.
int g_reuse_attempted = 0;
int g_reuse_observed  = 0;
struct ReuseCoverageReporter
{
    ~ReuseCoverageReporter()
    {
        if(g_reuse_attempted > 0 && g_reuse_observed == 0)
            std::cerr
                << "[TF32Reuse] WARNING: all " << g_reuse_attempted
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

    // The CK solver-type enum for this fixture's dimensionality and direction.
    static constexpr miopen::solver::CKSolverType CkSolver()
    {
        using CK = miopen::solver::CKSolverType;
        if constexpr(NDIM == 2u)
            return DIR == Direction::Forward        ? CK::GrpConvFwd
                   : DIR == Direction::BackwardData ? CK::GrpConvBwd
                                                    : CK::GrpConvWrw;
        else
            return DIR == Direction::Forward        ? CK::GrpConv3dFwd
                   : DIR == Direction::BackwardData ? CK::GrpConv3dBwd
                                                    : CK::GrpConv3dWrw;
    }

    // Set compute_type before the base SetUp() consumes it. Done here rather than in a constructor
    // to avoid cppcheck uninitDerivedMemberVar on the base fixture's members.
    void SetUp() override
    {
        compute_type = "TF32";
        Base::SetUp();
    }

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

    // Direction-specific invoke tensors and the host/device buffers holding this direction's
    // result. Mirrors GroupConvTestFix::DispatchSolver; its RunSolverImpl cannot be reused here
    // because it always uses the default perf config and never reads the result back for compare.
    auto MakeTensors()
    {
        if constexpr(DIR == Direction::Forward)
            return miopen::ConvDataTensors{
                input.desc, in_dev.get(), weights.desc, wei_dev.get(), output.desc, out_dev.get()};
        else if constexpr(DIR == Direction::BackwardData)
            return miopen::ConvDataTensors{
                output.desc, out_dev.get(), weights.desc, wei_dev.get(), input.desc, in_dev.get()};
        else
            return miopen::ConvWrwTensors{
                output.desc, out_dev.get(), input.desc, in_dev.get(), weights.desc, wei_dev.get()};
    }

    tensor<float>& ResultSrc()
    {
        if constexpr(DIR == Direction::Forward)
            return output;
        else if constexpr(DIR == Direction::BackwardData)
            return input;
        else
            return weights;
    }

    miopen::Allocator::ManageDataPtr& ResultDev()
    {
        if constexpr(DIR == Direction::Forward)
            return out_dev;
        else if constexpr(DIR == Direction::BackwardData)
            return in_dev;
        else
            return wei_dev;
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

        const auto tensors = MakeTensors();
        auto& result_src   = ResultSrc();
        auto& result_dev   = ResultDev();
        using InvokeParams = std::conditional_t<DIR == Direction::BackwardWeights,
                                                miopen::conv::WrWInvokeParams,
                                                miopen::conv::DataInvokeParams>;
        const auto invoke_params =
            InvokeParams{tensors, wspace.ptr(), wspace.size(), false, alpha, beta};

        // Run a prebuilt solution and read this direction's result back to host.
        auto invoke_sol = [&](auto sol, tensor<float>& result) {
            ASSERT_TRUE(sol.Succeeded());
            ASSERT_TRUE(sol.invoker_factory); // guard the deref below
            const auto invoker =
                handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
            invoker(handle, invoke_params);
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

        // Mirror the production restore path (find_solution.hpp): a reloaded config is validated
        // before use, so the restored (use_tf32-dropped) config must pass this gate.
        ASSERT_TRUE(solv.IsValidPerformanceConfig(ctx, problem_tf32, restored))
            << "restored perf-db config rejected by IsValidPerformanceConfig";

        // 2D bwd-data and all bwd-weights solvers store kernel_id as "<type>+<split_k>"; make sure
        // such a case is exercised, since GetSolutionTf32Aware must strip that suffix before the
        // TF32 membership test (a bare FillValidKernels entry never carries it).
        constexpr bool suffixed_id =
            DIR == Direction::BackwardWeights || (DIR == Direction::BackwardData && NDIM == 2u);
        if constexpr(suffixed_id)
            EXPECT_NE(restored.kernel_id.find('+'), std::string::npos)
                << "expected a split_k suffix on the kernel_id: " << restored.kernel_id;

        // References are built straight from the loader with an explicit compute type, so they do
        // not depend on the solver's GetSolution (the code under test) or on the config's dropped
        // use_tf32 flag. Using the restored kernel_id keeps every run on the same kernel.
        const auto& loader = miopen::solver::CkImplLibLoader::Get(handle.GetDeviceName());
        ASSERT_TRUE(loader.IsLoaded());

        tensor<float> out_fp32 = result_src;
        invoke_sol(
            loader.GetSolution(CkSolver(), ctx, problem_tf32, cfg.kernel_id, /*use_tf32=*/false),
            out_fp32);

        // A forced-TF32 build of the same id. If it cannot build, the id has no distinct TF32
        // instance here, so the reuse property is unobservable -- skip rather than false-fail.
        ++g_reuse_attempted;
        auto sol_tf32_ref =
            loader.GetSolution(CkSolver(), ctx, problem_tf32, cfg.kernel_id, /*use_tf32=*/true);
        if(!sol_tf32_ref.Succeeded())
        {
            test_skipped = true;
            GTEST_SKIP() << "no distinct TF32 instance for this kernel_id on this build/device";
        }
        tensor<float> out_tf32_ref = result_src;
        invoke_sol(std::move(sol_tf32_ref), out_tf32_ref);

        ASSERT_EQ(miopen::range_distance(out_tf32_ref), miopen::range_distance(out_fp32));

        // On some builds TF32 is numerically identical to FP32 for this instance; there the reuse
        // property is unobservable. Decide the skip from these code-under-test-independent runs.
        if(miopen::rms_range(out_fp32, out_tf32_ref) <= kDistinctTol)
        {
            test_skipped = true;
            GTEST_SKIP() << "CK runs TF32 identically to FP32 on this device; cannot observe reuse";
        }
        ++g_reuse_observed;

        // The fix under test: the solver must build the db-restored (use_tf32-dropped) config as
        // TF32 -- matching the forced-TF32 reference and diverging from FP32 -- not fall back to
        // FP32.
        tensor<float> out_restored = result_src;
        invoke_sol(solv.GetSolution(ctx, problem_tf32, restored), out_restored);

        EXPECT_GT(miopen::rms_range(out_fp32, out_restored), kDistinctTol)
            << "perf-db-restored TF32 config produced an FP32-identical result";
        EXPECT_LE(miopen::rms_range(out_tf32_ref, out_restored), kMatchTol)
            << "perf-db-restored TF32 result diverged from the forced-TF32 reference";
    }

    // FillValidKernels returns bare type strings, but the bwd-data/wrw solvers only accept a
    // kernel_id with a "+<split_k>" suffix (fwd needs none). Return the first candidate the loader
    // accepts under the requested compute mode -- bare if that is accepted, else with a split_k
    // suffix -- or "" if none is usable. Mirrors how a real perf-db id is formed for these solvers.
    std::string SupportedKernelId(const miopen::solver::CkImplLibLoader& loader,
                                  const miopen::conv::ProblemDescription& problem,
                                  const std::vector<std::string>& candidates,
                                  bool use_tf32)
    {
        for(const auto& bare : candidates)
        {
            if(loader.IsArgsSupported(CkSolver(), problem, bare, miopenFloat, use_tf32))
                return bare;
            for(int split_k : {1, 2, 4, 8, 16})
            {
                const auto id = bare + "+" + std::to_string(split_k);
                if(loader.IsArgsSupported(CkSolver(), problem, id, miopenFloat, use_tf32))
                    return id;
            }
        }
        return {};
    }

    // FillValidKernels(tf32=true) minus the fp32 set: kernel ids with a TF32 instance only.
    std::vector<std::string> KernelsOnlyIn(const std::vector<std::string>& set,
                                           const std::vector<std::string>& other)
    {
        std::vector<std::string> only;
        for(const auto& k : set)
            if(std::find(other.begin(), other.end(), k) == other.end())
                only.push_back(k);
        return only;
    }

    // Exercises the FP32-availability fallback in CkImplLibLoader::GetSolutionTf32Aware: a
    // kernel_id valid under FP32 but not TF32 (as chosen by FillValidKernelsWithTf32Fallback) must
    // build as FP32, not be forced into a missing TF32 instance. Skips on builds with no such
    // kernel.
    void RunFp32Fallback()
    {
        auto& handle = get_handle();
        auto ctx     = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        auto problem = MakeProblem(conv_desc);
        problem.SetupComputeType(ctx);
        if(!problem.UseTF32())
        {
            test_skipped = true;
            GTEST_SKIP() << "TF32 compute is not enabled on this device";
        }

        const auto& loader = miopen::solver::CkImplLibLoader::Get(handle.GetDeviceName());
        ASSERT_TRUE(loader.IsLoaded());
        const auto tf32 = loader.FillValidKernels(CkSolver(), problem, miopenFloat, true);
        const auto fp32 = loader.FillValidKernels(CkSolver(), problem, miopenFloat, false);
        const auto id   = SupportedKernelId(loader, problem, KernelsOnlyIn(fp32, tf32), false);
        if(id.empty())
        {
            test_skipped = true;
            GTEST_SKIP() << "no FP32-only grouped kernel on this build; fallback not reproducible";
        }

        // Precondition: forcing TF32 on this id fails to build (it has no TF32 instance).
        ASSERT_FALSE(loader.GetSolution(CkSolver(), ctx, problem, id, true).Succeeded())
            << "expected TF32 build of an FP32-only id to fail: " << id;
        // The fallback-aware path must recover and build it as FP32.
        EXPECT_TRUE(loader.GetSolutionTf32Aware(CkSolver(), ctx, problem, id).Succeeded())
            << "GetSolutionTf32Aware did not fall back to FP32 for an FP32-only id: " << id;
    }

    // Guards the IsValid fix: a perf-db-restored config whose kernel_id has a TF32 instance only
    // (use_tf32 dropped on restore) must still pass IsValidPerformanceConfig. Before the fix,
    // IsValid validated against the FP32 set with the stale use_tf32=false and dropped these
    // records. 2D bwd-data is the direction with TF32-only kernels; the others skip (no such id on
    // this shape).
    template <class Solver>
    void RunRestoreTf32OnlyId()
    {
        auto& handle = get_handle();
        auto ctx     = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        auto problem = MakeProblem(conv_desc);
        problem.SetupComputeType(ctx);
        if(!problem.UseTF32())
        {
            test_skipped = true;
            GTEST_SKIP() << "TF32 compute is not enabled on this device";
        }

        const auto& loader = miopen::solver::CkImplLibLoader::Get(handle.GetDeviceName());
        ASSERT_TRUE(loader.IsLoaded());
        const auto tf32 = loader.FillValidKernels(CkSolver(), problem, miopenFloat, true);
        const auto fp32 = loader.FillValidKernels(CkSolver(), problem, miopenFloat, false);
        const auto id   = SupportedKernelId(loader, problem, KernelsOnlyIn(tf32, fp32), true);
        if(id.empty())
        {
            test_skipped = true;
            GTEST_SKIP() << "no TF32-only kernel id on this build/device (expected except 2D bwd)";
        }

        // It is genuinely TF32-only: the FP32 instance set does not accept it.
        EXPECT_FALSE(loader.IsArgsSupported(CkSolver(), problem, id, miopenFloat, false))
            << "id expected to be TF32-only but is also supported as FP32: " << id;

        // A restored config carrying this id has use_tf32 dropped; IsValid must recompute TF32 and
        // still accept it, instead of validating against the FP32 set only.
        Solver solv{};
        auto proto = solv.GetDefaultPerformanceConfig(ctx, problem);
        std::decay_t<decltype(proto)> restored;
        ASSERT_TRUE(restored.Deserialize(id));
        ASSERT_EQ(restored.kernel_id, id);
        ASSERT_FALSE(restored.UseTF32()) << "test precondition: restored config must drop use_tf32";
        EXPECT_TRUE(solv.IsValidPerformanceConfig(ctx, problem, restored))
            << "restored TF32-only config wrongly rejected by IsValidPerformanceConfig: " << id;
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
// them. Every suite carries the numeric DbRestoredConfigStaysTF32 reuse test, the FP32-availability
// fallback (Fp32OnlyIdBuildsAsFp32), and the TF32-only-id IsValid guard
// (RestoreTf32OnlyIdValidates)
// -- the latter two skip on solvers/shapes without such a kernel (only 2D bwd-data has TF32-only
// ids here). TF32AndFp32GetDistinctDbKeys is solver-independent, so it is instantiated once.

struct GPU_GroupConv2D_Forward_TF32Reuse_FP32 : GroupConvTF32Reuse<2u, Direction::Forward>
{
};
TEST_P(GPU_GroupConv2D_Forward_TF32Reuse_FP32, DbRestoredConfigStaysTF32)
{
    RunReuse<miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops>();
}
TEST_P(GPU_GroupConv2D_Forward_TF32Reuse_FP32, Fp32OnlyIdBuildsAsFp32) { RunFp32Fallback(); }
TEST_P(GPU_GroupConv2D_Forward_TF32Reuse_FP32, RestoreTf32OnlyIdValidates)
{
    RunRestoreTf32OnlyId<miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops>();
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
TEST_P(GPU_GroupConv2D_BackwardData_TF32Reuse_FP32, Fp32OnlyIdBuildsAsFp32) { RunFp32Fallback(); }
TEST_P(GPU_GroupConv2D_BackwardData_TF32Reuse_FP32, RestoreTf32OnlyIdValidates)
{
    RunRestoreTf32OnlyId<miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops>();
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
TEST_P(GPU_GroupConv2D_BackwardWeights_TF32Reuse_FP32, Fp32OnlyIdBuildsAsFp32)
{
    RunFp32Fallback();
}
TEST_P(GPU_GroupConv2D_BackwardWeights_TF32Reuse_FP32, RestoreTf32OnlyIdValidates)
{
    RunRestoreTf32OnlyId<miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops>();
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
TEST_P(GPU_GroupConv3D_Forward_TF32Reuse_FP32, Fp32OnlyIdBuildsAsFp32) { RunFp32Fallback(); }
TEST_P(GPU_GroupConv3D_Forward_TF32Reuse_FP32, RestoreTf32OnlyIdValidates)
{
    RunRestoreTf32OnlyId<miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops>();
}
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
TEST_P(GPU_GroupConv3D_BackwardData_TF32Reuse_FP32, Fp32OnlyIdBuildsAsFp32) { RunFp32Fallback(); }
TEST_P(GPU_GroupConv3D_BackwardData_TF32Reuse_FP32, RestoreTf32OnlyIdValidates)
{
    RunRestoreTf32OnlyId<miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops>();
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
TEST_P(GPU_GroupConv3D_BackwardWeights_TF32Reuse_FP32, Fp32OnlyIdBuildsAsFp32)
{
    RunFp32Fallback();
}
TEST_P(GPU_GroupConv3D_BackwardWeights_TF32Reuse_FP32, RestoreTf32OnlyIdValidates)
{
    RunRestoreTf32OnlyId<miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops>();
}
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_GroupConv3D_BackwardWeights_TF32Reuse_FP32,
                         testing::Combine(testing::ValuesIn(Tf32ReuseConfigs<3u>()),
                                          testing::Values(1.0f),
                                          testing::Values(0.0f),
                                          testing::Values(Tf32ReuseLayout<3u>())));
