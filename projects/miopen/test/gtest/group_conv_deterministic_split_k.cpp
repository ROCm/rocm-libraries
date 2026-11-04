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

#include <gtest/gtest.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include "../random.hpp"
#include "get_handle.hpp"
#include "../driver/tensor_driver.hpp"
#include "conv_common.hpp"
#include "gtest_common.hpp"

namespace {

using Direction = miopen::conv::Direction;

// Small test configurations for fast execution - deterministic split_k tests
struct DeterministicSplitKTestConfig
{
    size_t G;
    size_t N;
    size_t C;
    size_t K;
    std::vector<size_t> img; // H, W for 2D or D, H, W for 3D
    std::vector<size_t> filter;
    std::vector<size_t> pad;
    std::vector<size_t> stride;
    std::vector<size_t> dilation;
    bool is_3d;
    bool nhwc; // NHWC (2D) or NDHWC (3D) instead of NCHW or NCDHW
    double alpha;
    double beta;

    friend std::ostream& operator<<(std::ostream& os, const DeterministicSplitKTestConfig& tc)
    {
        os << "G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " K:" << tc.K;
        if(tc.is_3d)
        {
            os << " D:" << tc.img[0] << " H:" << tc.img[1] << " W:" << tc.img[2];
            os << " z:" << tc.filter[0] << " y:" << tc.filter[1] << " x:" << tc.filter[2];
        }
        else
        {
            os << " H:" << tc.img[0] << " W:" << tc.img[1];
            os << " y:" << tc.filter[0] << " x:" << tc.filter[1];
        }
        os << (tc.nhwc ? " NHWC" : " NCHW") << " alpha:" << tc.alpha << " beta:" << tc.beta;
        return os;
    }
};

// Test configurations - small sizes for fast execution
std::vector<DeterministicSplitKTestConfig> GetDeterministicTestConfigs2D()
{
    return {// G  N  C   K   img     filter  pad   stride dilation 3d  nhwc  alpha beta
            {8, 1, 16, 16, {14, 14}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false, false, 1.0, 0.0},
            {4, 2, 8, 8, {16, 16}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false, false, 1.0, 0.0}};
}

std::vector<DeterministicSplitKTestConfig> GetDeterministicFwdTestConfigs2D()
{
    auto configs = GetDeterministicTestConfigs2D();
    // Large-N 1x3: a temporal conv1d with frames and spatial positions folded into the batch.
    configs.push_back({1, 256, 32, 32, {1, 64}, {1, 3}, {0, 1}, {1, 1}, {1, 1}, false});
    return configs;
}

std::vector<DeterministicSplitKTestConfig> GetDeterministicTestConfigs3D()
{
    return {// G  N  C  K  img        filter    pad       stride    dilation  3d  nhwc  alpha beta
            {4,
             1,
             8,
             8,
             {8, 14, 14},
             {3, 3, 3},
             {1, 1, 1},
             {1, 1, 1},
             {1, 1, 1},
             true,
             false,
             1.0,
             0.0},
            {2,
             2,
             4,
             4,
             {6, 12, 12},
             {2, 2, 2},
             {1, 1, 1},
             {1, 1, 1},
             {1, 1, 1},
             true,
             false,
             1.0,
             0.0}};
}

// ConvDirectNaiveConvWrw tiles the FP32 reduction when N*Do*Ho*Wo > 262144. These
// configurations are above that threshold.
std::vector<DeterministicSplitKTestConfig> GetDeterministicNaiveWrwTestConfigs2D()
{
    return {// G  N  C  K  img         filter  pad     stride  dilation 3d  nhwc  alpha beta
            {1, 2, 4, 4, {368, 368}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false, false, 1.0, 0.0},
            {2, 2, 4, 4, {368, 368}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false, false, 1.0, 0.0},
            // N*Ho*Wo = 541696: 2048 tiles (the limit) of 265 positions; tiles 2045-2047 are
            // empty.
            {1, 4, 4, 4, {368, 368}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false, false, 1.0, 0.0},
            {1, 2, 4, 4, {368, 368}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false, true, 1.0, 0.0},
            {1, 2, 4, 4, {368, 368}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false, false, 0.5, 0.75}};
}

std::vector<DeterministicSplitKTestConfig> GetDeterministicNaiveWrwTestConfigs3D()
{
    return {// G  N  C  K  img            filter     pad        stride     dilation   3d  nhwc alpha
            // beta
            {1,
             1,
             4,
             4,
             {8, 184, 184},
             {3, 3, 3},
             {1, 1, 1},
             {1, 1, 1},
             {1, 1, 1},
             true,
             false,
             1.0,
             0.0},
            {1,
             1,
             4,
             4,
             {8, 184, 184},
             {3, 3, 3},
             {1, 1, 1},
             {1, 1, 1},
             {1, 1, 1},
             true,
             true,
             1.0,
             0.0}};
}

template <typename T, Direction CONV_DIR, typename SolverType, bool IS_3D>
class GPU_GroupConvDeterministicSplitK
    : public ::testing::TestWithParam<DeterministicSplitKTestConfig>
{
protected:
    static constexpr int NUM_ITERATIONS = 10;

    void SetUp() override { prng::reset_seed(); }

    void RunTest()
    {
        const auto& config = GetParam();
        std::cout << "Testing configuration: " << config << std::endl;

        auto& handle = get_handle();

        // Create tensors with appropriate layout
        miopenTensorLayout_t layout = IS_3D ? (config.nhwc ? miopenTensorNDHWC : miopenTensorNCDHW)
                                            : (config.nhwc ? miopenTensorNHWC : miopenTensorNCHW);

        std::vector<size_t> input_dims  = {config.N, config.C};
        std::vector<size_t> weight_dims = {config.K, config.C / config.G};
        input_dims.insert(input_dims.end(), config.img.begin(), config.img.end());
        weight_dims.insert(weight_dims.end(), config.filter.begin(), config.filter.end());

        tensor<T> input{layout, input_dims};
        tensor<T> weights{layout, weight_dims};

        // Create convolution descriptor with deterministic mode enabled
        auto conv_desc = miopen::ConvolutionDescriptor{
            static_cast<int>(config.pad.size()),
            miopenConvolution,
            miopenPaddingDefault,
            std::vector<int>(config.pad.begin(), config.pad.end()),
            std::vector<int>(config.stride.begin(), config.stride.end()),
            std::vector<int>(config.dilation.begin(), config.dilation.end()),
            std::vector<int>(config.pad.size(), 0),
            static_cast<int>(config.G),
            1.0};

        // Enable deterministic mode
        conv_desc.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
        ASSERT_TRUE(conv_desc.attribute.deterministic.Get() == 1);

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        tensor<T> output{layout, output_desc.GetLengths()};

        // Initialize tensors
        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };

        if constexpr(CONV_DIR == Direction::Forward)
        {
            input.generate(gen_value);
            weights.generate(gen_value);
            std::fill(output.begin(), output.end(), T(0));
        }
        else if constexpr(CONV_DIR == Direction::BackwardData)
        {
            output.generate(gen_value);
            weights.generate(gen_value);
            std::fill(input.begin(), input.end(), T(0));
        }
        else // BackwardWeights
        {
            input.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.1), static_cast<T>(0.1));
            });
            output.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.01), static_cast<T>(0.1));
            });
            // With beta != 0 the result also depends on the initial weights.
            if(config.beta != 0.0)
                weights.generate(gen_value);
            else
                std::fill(weights.begin(), weights.end(), T{0});
        }
        const std::vector<T> wei_init = weights.data;

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(output.data);

        // Create solver and problem
        SolverType solv{};
        auto ctx = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        miopen::conv::ProblemDescription problem = [&]() {
            if constexpr(CONV_DIR == Direction::Forward)
                return miopen::conv::ProblemDescription{
                    input.desc, weights.desc, output.desc, conv_desc, CONV_DIR};
            else
                return miopen::conv::ProblemDescription{
                    output.desc, weights.desc, input.desc, conv_desc, CONV_DIR};
        }();

        if(!solv.IsApplicable(ctx, problem))
        {
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable";
        }

        std::cout << "Using solver: " << solv.SolverDbId() << std::endl;

        Workspace wspace{};
        if(solv.MayNeedWorkspace())
        {
            wspace.resize(solv.GetWorkspaceSize(ctx, problem));
        }

        miopen::solver::ConvSolution sol;
        if constexpr(miopen::solver::IsTunable<SolverType>::value)
        {
            auto perf_config = solv.GetDefaultPerformanceConfig(ctx, problem);
            std::cout << "Performance config: " << perf_config << std::endl;
            sol = solv.GetSolution(ctx, problem, perf_config);
        }
        else
        {
            sol = solv.GetSolution(ctx, problem);
        }
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);

        // Store results from each iteration
        std::vector<std::vector<T>> results;
        results.reserve(NUM_ITERATIONS);

        for(int i = 0; i < NUM_ITERATIONS; ++i)
        {
            // Reset output buffer
            if constexpr(CONV_DIR == Direction::Forward)
            {
                std::fill(output.begin(), output.end(), T(0));
                out_dev = handle.Write(output.data);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                std::fill(input.begin(), input.end(), T(0));
                in_dev = handle.Write(input.data);
            }
            else
            {
                weights.data = wei_init;
                wei_dev      = handle.Write(weights.data);
            }

            // Execute kernel
            if constexpr(CONV_DIR == Direction::Forward)
            {
                auto invoke_params =
                    miopen::conv::DataInvokeParams{miopen::ConvFwdTensors{input.desc,
                                                                          in_dev.get(),
                                                                          weights.desc,
                                                                          wei_dev.get(),
                                                                          output.desc,
                                                                          out_dev.get()},
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   false};
                (invoker)(handle, invoke_params);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                auto invoke_params =
                    miopen::conv::DataInvokeParams{miopen::ConvDataTensors{output.desc,
                                                                           out_dev.get(),
                                                                           weights.desc,
                                                                           wei_dev.get(),
                                                                           input.desc,
                                                                           in_dev.get()},
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   false};
                (invoker)(handle, invoke_params);
            }
            else
            {
                auto invoke_params =
                    miopen::conv::WrWInvokeParams{miopen::ConvWrwTensors{output.desc,
                                                                         out_dev.get(),
                                                                         input.desc,
                                                                         in_dev.get(),
                                                                         weights.desc,
                                                                         wei_dev.get()},
                                                  wspace.ptr(),
                                                  wspace.size(),
                                                  false,
                                                  miopen::Scalar(config.alpha),
                                                  miopen::Scalar(config.beta)};
                (invoker)(handle, invoke_params);
            }

            handle.Finish();

            // Read results
            if constexpr(CONV_DIR == Direction::Forward)
            {
                handle.ReadToVec(out_dev, output.data);
                results.push_back(output.data);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                handle.ReadToVec(in_dev, input.data);
                results.push_back(input.data);
            }
            else
            {
                handle.ReadToVec(wei_dev, weights.data);
                results.push_back(weights.data);
            }
        }

        // Verify bit-exact determinism
        const auto& reference = results[0];
        for(int i = 1; i < NUM_ITERATIONS; ++i)
        {
            const auto& current = results[i];
            ASSERT_EQ(reference.size(), current.size());

            bool match            = true;
            size_t first_mismatch = 0;
            for(size_t j = 0; j < reference.size(); ++j)
            {
                if(std::memcmp(&reference[j], &current[j], sizeof(T)) != 0)
                {
                    match          = false;
                    first_mismatch = j;
                    break;
                }
            }

            ASSERT_TRUE(match) << "Bit-exact mismatch at iteration " << i << ", element "
                               << first_mismatch << ": reference = " << reference[first_mismatch]
                               << ", current = " << current[first_mismatch];
        }

        std::cout << "✓ All " << NUM_ITERATIONS << " iterations produced bit-exact results"
                  << std::endl;

        // The deterministic naive WRW path adds tile sums from a workspace. A wrong slice or
        // a missing tile is reproducible, so also compare with the reference.
        if constexpr(std::is_same_v<SolverType, miopen::solver::conv::ConvDirectNaiveConvWrw>)
        {
            // ref_conv_wrw ignores the initial weights, so add the alpha and beta terms here.
            auto ref = ref_conv_wrw(input, weights, output, conv_desc);
            for(size_t j = 0; j < ref.data.size(); ++j)
                ref.data[j] =
                    static_cast<T>(config.alpha * ref.data[j] + config.beta * wei_init[j]);
            ASSERT_FALSE(miopen::range_zero(weights)) << "GPU data is all zeros";
            // 80 * eps is the limit of the WRW solver tests (solver_wrw.hpp). The sum here is two
            // fixed fp32 chains (at most 2048 tile sums of at most 265 terms). Its random rounding
            // error grows as about sqrt(2048 + 265) * eps = 48 * eps, which is less than 80 * eps.
            const double threshold = std::numeric_limits<T>::epsilon() * 80;
            const auto error       = miopen::rms_range(ref, weights);
            EXPECT_LT(error, threshold) << "Error beyond tolerance";
        }
    }
};

} // namespace

// 2D FWD
using GPU_GroupConv2D_Deterministic_FWD_FP32 =
    GPU_GroupConvDeterministicSplitK<float,
                                     Direction::Forward,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops,
                                     false>;
using GPU_GroupConv2D_Deterministic_FWD_FP16 =
    GPU_GroupConvDeterministicSplitK<half,
                                     Direction::Forward,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops,
                                     false>;
using GPU_GroupConv2D_Deterministic_FWD_BFP16 =
    GPU_GroupConvDeterministicSplitK<bfloat16,
                                     Direction::Forward,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops,
                                     false>;

// 3D FWD
using GPU_GroupConv3D_Deterministic_FWD_FP32 =
    GPU_GroupConvDeterministicSplitK<float,
                                     Direction::Forward,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops,
                                     true>;
using GPU_GroupConv3D_Deterministic_FWD_FP16 =
    GPU_GroupConvDeterministicSplitK<half,
                                     Direction::Forward,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops,
                                     true>;
using GPU_GroupConv3D_Deterministic_FWD_BFP16 =
    GPU_GroupConvDeterministicSplitK<bfloat16,
                                     Direction::Forward,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops,
                                     true>;

// 3D BWD
using GPU_GroupConv3D_Deterministic_BWD_FP32 =
    GPU_GroupConvDeterministicSplitK<float,
                                     Direction::BackwardData,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops,
                                     true>;
using GPU_GroupConv3D_Deterministic_BWD_FP16 =
    GPU_GroupConvDeterministicSplitK<half,
                                     Direction::BackwardData,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops,
                                     true>;
using GPU_GroupConv3D_Deterministic_BWD_BFP16 =
    GPU_GroupConvDeterministicSplitK<bfloat16,
                                     Direction::BackwardData,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops,
                                     true>;

// 2D BWD
using GPU_GroupConv2D_Deterministic_BWD_FP32 =
    GPU_GroupConvDeterministicSplitK<float,
                                     Direction::BackwardData,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops,
                                     false>;
using GPU_GroupConv2D_Deterministic_BWD_FP16 =
    GPU_GroupConvDeterministicSplitK<half,
                                     Direction::BackwardData,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops,
                                     false>;
using GPU_GroupConv2D_Deterministic_BWD_BFP16 =
    GPU_GroupConvDeterministicSplitK<bfloat16,
                                     Direction::BackwardData,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops,
                                     false>;

// 2D WRW
using GPU_GroupConv2D_Deterministic_WRW_FP32 =
    GPU_GroupConvDeterministicSplitK<float,
                                     Direction::BackwardWeights,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops,
                                     false>;
using GPU_GroupConv2D_Deterministic_WRW_FP16 =
    GPU_GroupConvDeterministicSplitK<half,
                                     Direction::BackwardWeights,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops,
                                     false>;
using GPU_GroupConv2D_Deterministic_WRW_BFP16 =
    GPU_GroupConvDeterministicSplitK<bfloat16,
                                     Direction::BackwardWeights,
                                     miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops,
                                     false>;

// 3D WRW
using GPU_GroupConv3D_Deterministic_WRW_FP32 =
    GPU_GroupConvDeterministicSplitK<float,
                                     Direction::BackwardWeights,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops,
                                     true>;
using GPU_GroupConv3D_Deterministic_WRW_FP16 =
    GPU_GroupConvDeterministicSplitK<half,
                                     Direction::BackwardWeights,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops,
                                     true>;
using GPU_GroupConv3D_Deterministic_WRW_BFP16 =
    GPU_GroupConvDeterministicSplitK<bfloat16,
                                     Direction::BackwardWeights,
                                     miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops,
                                     true>;

// Naive WRW (FP32 only: the solver does not tile other data types)
using GPU_GroupConv2D_Deterministic_NaiveWRW_FP32 =
    GPU_GroupConvDeterministicSplitK<float,
                                     Direction::BackwardWeights,
                                     miopen::solver::conv::ConvDirectNaiveConvWrw,
                                     false>;
using GPU_GroupConv3D_Deterministic_NaiveWRW_FP32 =
    GPU_GroupConvDeterministicSplitK<float,
                                     Direction::BackwardWeights,
                                     miopen::solver::conv::ConvDirectNaiveConvWrw,
                                     true>;

// Test definitions
TEST_P(GPU_GroupConv2D_Deterministic_FWD_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_FWD_FP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_FWD_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_FWD_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_FWD_FP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_FWD_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_BWD_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_BWD_FP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_BWD_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_BWD_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_BWD_FP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_BWD_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_WRW_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_WRW_FP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_WRW_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_WRW_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_WRW_FP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_WRW_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv2D_Deterministic_NaiveWRW_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_GroupConv3D_Deterministic_NaiveWRW_FP32, DeterministicTest) { this->RunTest(); };

// Test instantiations
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_FWD_FP32,
                         testing::ValuesIn(GetDeterministicFwdTestConfigs2D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_FWD_FP16,
                         testing::ValuesIn(GetDeterministicFwdTestConfigs2D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_FWD_BFP16,
                         testing::ValuesIn(GetDeterministicFwdTestConfigs2D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_FWD_FP32,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_FWD_FP16,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_FWD_BFP16,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_BWD_FP32,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_BWD_FP16,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_BWD_BFP16,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_BWD_FP32,
                         testing::ValuesIn(GetDeterministicTestConfigs2D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_BWD_FP16,
                         testing::ValuesIn(GetDeterministicTestConfigs2D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_BWD_BFP16,
                         testing::ValuesIn(GetDeterministicTestConfigs2D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_WRW_FP32,
                         testing::ValuesIn(GetDeterministicTestConfigs2D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_WRW_FP16,
                         testing::ValuesIn(GetDeterministicTestConfigs2D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_WRW_BFP16,
                         testing::ValuesIn(GetDeterministicTestConfigs2D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_WRW_FP32,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_WRW_FP16,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_WRW_BFP16,
                         testing::ValuesIn(GetDeterministicTestConfigs3D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv2D_Deterministic_NaiveWRW_FP32,
                         testing::ValuesIn(GetDeterministicNaiveWrwTestConfigs2D()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_GroupConv3D_Deterministic_NaiveWRW_FP32,
                         testing::ValuesIn(GetDeterministicNaiveWrwTestConfigs3D()));
