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

// Regression tests for the TF32 perf-db reuse bug: a CK grouped-conv PerformanceConfig only
// serializes its kernel_id, so a config restored from the perf-db loses its use_tf32 flag.
// GetSolution used to trust that (now false) flag and silently built the FP32 instance -- same
// kernel name, FP32 math -- for a TF32 problem. The fix recomputes use_tf32 from the problem
// (whose key already carries the _cxTF32 compute marker) at solution-build time.

#include <gtest/gtest.h>

#include "group_conv.hpp"

#include <miopen/conv/solvers.hpp>
#include <miopen/tensor.hpp>

#include <sstream>
#include <string>
#include <type_traits>

using namespace group_conv;

// ---------------------------------------------------------------------------
// GPU: run the forward grouped CK solver with a config that has been round-tripped through the
// perf-db serialization (which drops use_tf32), on a TF32 problem, and require the result to
// differ from the same kernel run as strict FP32. Before the fix the DB-restored config built
// the FP32 instance for the TF32 problem, giving a byte-identical FP32 result; after the fix
// GetSolution recomputes use_tf32 from the problem and the TF32 kernel runs, diverging from FP32.
// ---------------------------------------------------------------------------
struct GPU_GroupConv2DFwd_TF32Reuse_FP32 : GroupConvTestFix<2u, float, Direction::Forward>
{
    // compute_type must be set before the base SetUp() consumes it; do it here rather than in a
    // constructor so the fixture needs no user-declared constructor.
    void SetUp() override
    {
        compute_type = "TF32";
        GroupConvTestFix<2u, float, Direction::Forward>::SetUp();
    }

    // This test runs the solver itself and does its own verification.
    void TearDown() override {}

    void RunDbReuse()
    {
        using Solver = miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops;

        auto& handle = get_handle();
        Solver solv{};
        auto ctx = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        // conv_desc already has math_type == Default (SetUp set it for compute_type == "TF32").
        miopen::conv::ProblemDescription problem_tf32{input.desc,
                                                      weights.desc,
                                                      output.desc,
                                                      conv_desc,
                                                      Direction::Forward,
                                                      0 /*bias*/,
                                                      alpha,
                                                      beta};
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

        // Same shape forced to strict FP32 -- the ground truth to diff against.
        auto conv_desc_fp32 = conv_desc;
        conv_desc_fp32.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathPedantic);
        miopen::conv::ProblemDescription problem_fp32{input.desc,
                                                      weights.desc,
                                                      output.desc,
                                                      conv_desc_fp32,
                                                      Direction::Forward,
                                                      0 /*bias*/,
                                                      alpha,
                                                      beta};

        if(solv.MayNeedWorkspace())
        {
            wspace.resize(std::max(solv.GetWorkspaceSize(ctx, problem_tf32),
                                   solv.GetWorkspaceSize(ctx, problem_fp32)));
        }

        const miopen::ConvDataTensors tensors{
            input.desc, in_dev.get(), weights.desc, wei_dev.get(), output.desc, out_dev.get()};
        const auto invoke_params = miopen::conv::DataInvokeParams{
            tensors, wspace.ptr(), wspace.size(), false, alpha, beta};

        auto run = [&](const miopen::conv::ProblemDescription& prob,
                       const auto& perf_cfg,
                       tensor<float>& result) {
            auto sol = solv.GetSolution(ctx, prob, perf_cfg);
            EXPECT_TRUE(sol.Succeeded());
            EXPECT_TRUE(sol.invoker_factory);
            const auto invoker =
                handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
            (invoker)(handle, invoke_params);
            handle.Finish();
            handle.ReadToVec(out_dev, result.data);
        };

        // Heuristic config for the TF32 problem, then round-trip it through the perf-db
        // serialization -- this keeps only kernel_id and clears use_tf32, exactly the state a
        // config is in after a perf-db reload.
        auto cfg = solv.GetDefaultPerformanceConfig(ctx, problem_tf32);
        std::ostringstream oss;
        cfg.Serialize(oss);
        std::decay_t<decltype(cfg)> restored;
        ASSERT_TRUE(restored.Deserialize(oss.str()));
        ASSERT_EQ(restored.kernel_id, cfg.kernel_id);
        ASSERT_FALSE(restored.UseTF32())
            << "test precondition: a reloaded perf-db config must have dropped use_tf32";

        tensor<float> out_tf32 = output;
        run(problem_tf32, restored, out_tf32);
        tensor<float> out_fp32 = output;
        run(problem_fp32, restored, out_fp32);

        // Pristine TF32 config (use_tf32 preserved, no db round-trip): the reference for what a
        // genuinely-TF32 run looks like on this build.
        tensor<float> out_tf32_ref = output;
        run(problem_tf32, cfg, out_tf32_ref);

        ASSERT_EQ(miopen::range_distance(out_tf32), miopen::range_distance(out_fp32));

        // Some CK builds do not carry a numerically distinct TF32 grouped-fwd kernel for this
        // instance on this device -- there TF32 and FP32 compute identically and the perf-db reuse
        // property is unobservable. Skip rather than false-fail; the fix is still exercised on
        // builds that do run a distinct TF32 kernel (e.g. gfx942 / ROCm >= 10.1).
        if(miopen::rms_range(out_fp32, out_tf32_ref) <= 1e-5)
        {
            test_skipped = true;
            GTEST_SKIP() << "CK runs TF32 grouped-fwd identically to FP32 on this device; cannot "
                            "observe perf-db TF32 reuse";
        }

        // The fix: a db-restored config (use_tf32 dropped) must still run TF32 -- matching the
        // pristine TF32 reference and diverging from FP32 -- not silently fall back to FP32.
        EXPECT_GT(miopen::rms_range(out_fp32, out_tf32), 1e-5)
            << "perf-db-restored TF32 config produced an FP32-identical result: TF32 silently fell "
               "back to FP32 on reuse";
        EXPECT_LE(miopen::rms_range(out_tf32_ref, out_tf32), 1e-6)
            << "perf-db-restored TF32 result diverged from the pristine TF32 result";
    }

    // TF32 and strict-FP32 versions of the same problem must serialize to different perf-db keys
    // (via the _cxTF32 compute marker) so their tuned records never collide in the DB. The marker
    // is only emitted once SetupComputeType has run against a TF32-capable device, so this runs on
    // the GPU and skips elsewhere.
    void RunKeyDistinct()
    {
        auto& handle = get_handle();
        auto ctx     = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        miopen::conv::ProblemDescription problem_tf32{
            input.desc, weights.desc, output.desc, conv_desc, Direction::Forward, 0, alpha, beta};
        problem_tf32.SetupComputeType(ctx);
        if(!problem_tf32.UseTF32())
        {
            test_skipped = true;
            GTEST_SKIP() << "TF32 compute is not enabled on this device";
        }

        auto conv_desc_fp32 = conv_desc;
        conv_desc_fp32.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathPedantic);
        miopen::conv::ProblemDescription problem_fp32{input.desc,
                                                      weights.desc,
                                                      output.desc,
                                                      conv_desc_fp32,
                                                      Direction::Forward,
                                                      0,
                                                      alpha,
                                                      beta};
        problem_fp32.SetupComputeType(ctx);

        std::ostringstream key_tf32;
        std::ostringstream key_fp32;
        problem_tf32.Serialize(key_tf32);
        problem_fp32.Serialize(key_fp32);

        EXPECT_NE(key_tf32.str().find("cxTF32"), std::string::npos)
            << "TF32 problem key is missing the compute marker: " << key_tf32.str();
        EXPECT_EQ(key_fp32.str().find("cxTF32"), std::string::npos)
            << "strict-FP32 problem key unexpectedly carries the TF32 marker: " << key_fp32.str();
        EXPECT_NE(key_tf32.str(), key_fp32.str())
            << "TF32 and FP32 problems serialize to the same perf-db key; their records collide";
    }
};

TEST_P(GPU_GroupConv2DFwd_TF32Reuse_FP32, DbRestoredConfigStaysTF32) { RunDbReuse(); }
TEST_P(GPU_GroupConv2DFwd_TF32Reuse_FP32, TF32AndFp32GetDistinctDbKeys) { RunKeyDistinct(); }

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_GroupConv2DFwd_TF32Reuse_FP32,
    testing::Combine(testing::Values(GroupConvTestConfig<2u>{
                         8, 256, 384, 384, {28, 28}, {3, 3}, {1, 1}, {1, 1}, {1, 1}}),
                     testing::Values(1.0f),
                     testing::Values(0.0f),
                     testing::Values(miopenTensorNHWC)));
