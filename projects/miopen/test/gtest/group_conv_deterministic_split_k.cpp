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

namespace group_conv_deterministic {

using Direction = miopen::conv::Direction;

// Small test configurations for fast execution
struct DeterministicTestConfig2D
{
    size_t G;
    size_t N;
    size_t C;
    size_t K;
    size_t H;
    size_t W;
    size_t y; // filter height
    size_t x; // filter width
    size_t pad_h;
    size_t pad_w;
    size_t stride_h;
    size_t stride_w;
    size_t dilation_h;
    size_t dilation_w;

    std::vector<size_t> GetInput() const { return {N, C, H, W}; }

    std::vector<size_t> GetWeights() const { return {K, C / G, y, x}; }

    miopen::ConvolutionDescriptor GetConv() const
    {
        auto conv = miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad_h), static_cast<int>(pad_w)},
            {static_cast<int>(stride_h), static_cast<int>(stride_w)},
            {static_cast<int>(dilation_h), static_cast<int>(dilation_w)},
            {0, 0},
            static_cast<int>(G),
            1.0};

        // Enable deterministic mode
        conv.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);

        return conv;
    }

    friend std::ostream& operator<<(std::ostream& os, const DeterministicTestConfig2D& tc)
    {
        return os << "G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " K:" << tc.K
                  << " H:" << tc.H << " W:" << tc.W << " y:" << tc.y << " x:" << tc.x
                  << " pad_h:" << tc.pad_h << " pad_w:" << tc.pad_w << " stride_h:" << tc.stride_h
                  << " stride_w:" << tc.stride_w << " dilation_h:" << tc.dilation_h
                  << " dilation_w:" << tc.dilation_w;
    }
};

struct DeterministicTestConfig3D
{
    size_t G;
    size_t N;
    size_t C;
    size_t K;
    size_t D;
    size_t H;
    size_t W;
    size_t z; // filter depth
    size_t y; // filter height
    size_t x; // filter width
    size_t pad_d;
    size_t pad_h;
    size_t pad_w;
    size_t stride_d;
    size_t stride_h;
    size_t stride_w;
    size_t dilation_d;
    size_t dilation_h;
    size_t dilation_w;

    std::vector<size_t> GetInput() const { return {N, C, D, H, W}; }

    std::vector<size_t> GetWeights() const { return {K, C / G, z, y, x}; }

    miopen::ConvolutionDescriptor GetConv() const
    {
        auto conv = miopen::ConvolutionDescriptor{
            3,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad_d), static_cast<int>(pad_h), static_cast<int>(pad_w)},
            {static_cast<int>(stride_d), static_cast<int>(stride_h), static_cast<int>(stride_w)},
            {static_cast<int>(dilation_d),
             static_cast<int>(dilation_h),
             static_cast<int>(dilation_w)},
            {0, 0, 0},
            static_cast<int>(G),
            1.0};

        // Enable deterministic mode
        conv.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);

        return conv;
    }

    friend std::ostream& operator<<(std::ostream& os, const DeterministicTestConfig3D& tc)
    {
        return os << "G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " K:" << tc.K
                  << " D:" << tc.D << " H:" << tc.H << " W:" << tc.W << " z:" << tc.z
                  << " y:" << tc.y << " x:" << tc.x << " pad_d:" << tc.pad_d
                  << " pad_h:" << tc.pad_h << " pad_w:" << tc.pad_w << " stride_d:" << tc.stride_d
                  << " stride_h:" << tc.stride_h << " stride_w:" << tc.stride_w
                  << " dilation_d:" << tc.dilation_d << " dilation_h:" << tc.dilation_h
                  << " dilation_w:" << tc.dilation_w;
    }
};

template <typename T, typename ConfigType, Direction CONV_DIR, typename SolverType>
class DeterministicTest : public ::testing::Test
{
protected:
    static constexpr int NUM_ITERATIONS = 10;

    void RunDeterministicTest(const ConfigType& config)
    {
        std::cout << "Testing configuration: " << config << std::endl;

        auto& handle = get_handle();

        // Create tensors with appropriate layout based on dimensionality
        miopenTensorLayout_t layout;
        if constexpr(std::is_same_v<ConfigType, DeterministicTestConfig2D>)
        {
            layout = miopenTensorNCHW; // Use NCHW for 2D for better determinism
        }
        else
        {
            layout = miopenTensorNCDHW; // Use NCDHW for 3D
        }

        tensor<T> input{layout, config.GetInput()};
        tensor<T> weights{layout, config.GetWeights()};

        auto conv_desc = config.GetConv();

        // Verify deterministic mode is enabled
        ASSERT_TRUE(conv_desc.attribute.deterministic.Get() == 1);

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        tensor<T> output{layout, output_desc.GetLengths()};

        // Initialize input tensors with random data
        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };

        if constexpr(CONV_DIR == Direction::BackwardData)
        {
            output.generate(gen_value);
            weights.generate(gen_value);
            std::fill(input.begin(), input.end(), T(0));
        }
        else if constexpr(CONV_DIR == Direction::BackwardWeights)
        {
            input.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.1), static_cast<T>(0.1));
            });
            output.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.01), static_cast<T>(0.1));
            });
            std::fill(weights.begin(), weights.end(), T{0});
        }

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(output.data);

        // Create solver and problem description
        SolverType solv{};
        auto ctx = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        miopen::conv::ProblemDescription problem;
        if constexpr(CONV_DIR == Direction::BackwardData)
        {
            problem = miopen::conv::ProblemDescription{
                output.desc, weights.desc, input.desc, conv_desc, CONV_DIR};
        }
        else // BackwardWeights
        {
            problem = miopen::conv::ProblemDescription{
                output.desc, weights.desc, input.desc, conv_desc, CONV_DIR};
        }

        if(!solv.IsApplicable(ctx, problem))
        {
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable for this problem";
        }

        std::cout << "Using solver: " << solv.SolverDbId() << std::endl;

        // Allocate workspace if needed
        Workspace wspace{};
        if(solv.MayNeedWorkspace())
        {
            wspace.resize(solv.GetWorkspaceSize(ctx, problem));
        }

        // Get solution and create invoker
        auto perf_config = solv.GetDefaultPerformanceConfig(ctx, problem);
        auto sol         = solv.GetSolution(ctx, problem, perf_config);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);

        // Log the performance config to verify split_k and other parameters
        std::cout << "Performance config: " << perf_config << std::endl;

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);

        // Store results from each iteration
        std::vector<std::vector<T>> results;
        results.reserve(NUM_ITERATIONS);

        // Run the solver NUM_ITERATIONS times
        for(int i = 0; i < NUM_ITERATIONS; ++i)
        {
            // Reset output buffer
            if constexpr(CONV_DIR == Direction::BackwardData)
            {
                std::fill(input.begin(), input.end(), T(0));
                in_dev = handle.Write(input.data);
            }
            else // BackwardWeights
            {
                std::fill(weights.begin(), weights.end(), T(0));
                wei_dev = handle.Write(weights.data);
            }

            // Execute the kernel
            if constexpr(CONV_DIR == Direction::BackwardData)
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
            else // BackwardWeights
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
                                                  false};
                (invoker)(handle, invoke_params);
            }

            handle.Finish();

            // Read back results
            if constexpr(CONV_DIR == Direction::BackwardData)
            {
                handle.ReadToVec(in_dev, input.data);
                results.push_back(input.data);
            }
            else // BackwardWeights
            {
                handle.ReadToVec(wei_dev, weights.data);
                results.push_back(weights.data);
            }
        }

        // Verify all results are bit-exact
        const auto& reference = results[0];

        for(int i = 1; i < NUM_ITERATIONS; ++i)
        {
            const auto& current = results[i];

            ASSERT_EQ(reference.size(), current.size()) << "Size mismatch at iteration " << i;

            // Perform bit-exact comparison
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

            ASSERT_TRUE(match) << "Bit-exact mismatch found at iteration " << i << ", element "
                               << first_mismatch << ": reference = " << reference[first_mismatch]
                               << ", current = " << current[first_mismatch];
        }
    }
};

// 2D BWD Tests
using DeterministicTest2D_BWD_FP32 =
    DeterministicTest<float,
                      DeterministicTestConfig2D,
                      Direction::BackwardData,
                      miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops>;

using DeterministicTest2D_BWD_FP16 =
    DeterministicTest<half,
                      DeterministicTestConfig2D,
                      Direction::BackwardData,
                      miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops>;

using DeterministicTest2D_BWD_BFP16 =
    DeterministicTest<bfloat16,
                      DeterministicTestConfig2D,
                      Direction::BackwardData,
                      miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops>;

// 2D WRW Tests
using DeterministicTest2D_WRW_FP32 =
    DeterministicTest<float,
                      DeterministicTestConfig2D,
                      Direction::BackwardWeights,
                      miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops>;

using DeterministicTest2D_WRW_FP16 =
    DeterministicTest<half,
                      DeterministicTestConfig2D,
                      Direction::BackwardWeights,
                      miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops>;

using DeterministicTest2D_WRW_BFP16 =
    DeterministicTest<bfloat16,
                      DeterministicTestConfig2D,
                      Direction::BackwardWeights,
                      miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops>;

// 3D WRW Tests
using DeterministicTest3D_WRW_FP32 =
    DeterministicTest<float,
                      DeterministicTestConfig3D,
                      Direction::BackwardWeights,
                      miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops>;

using DeterministicTest3D_WRW_FP16 =
    DeterministicTest<half,
                      DeterministicTestConfig3D,
                      Direction::BackwardWeights,
                      miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops>;

using DeterministicTest3D_WRW_BFP16 =
    DeterministicTest<bfloat16,
                      DeterministicTestConfig3D,
                      Direction::BackwardWeights,
                      miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops>;

// 2D BWD Test Cases
TEST_F(DeterministicTest2D_BWD_FP32, SmallConfig1)
{
    // g   n   C   K   H   W   y  x  pad_h pad_w stride_h stride_w dilation_h dilation_w
    RunDeterministicTest({8, 1, 16, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest2D_BWD_FP16, SmallConfig1)
{
    RunDeterministicTest({8, 1, 16, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest2D_BWD_BFP16, SmallConfig1)
{
    RunDeterministicTest({8, 1, 16, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest2D_BWD_FP32, SmallConfig2)
{
    RunDeterministicTest({4, 2, 8, 8, 16, 16, 2, 2, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest2D_BWD_FP16, SmallConfig2)
{
    RunDeterministicTest({4, 2, 8, 8, 16, 16, 2, 2, 1, 1, 1, 1, 1, 1});
}

// 2D WRW Test Cases
TEST_F(DeterministicTest2D_WRW_FP32, SmallConfig1)
{
    RunDeterministicTest({8, 1, 16, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest2D_WRW_FP16, SmallConfig1)
{
    RunDeterministicTest({8, 1, 16, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest2D_WRW_BFP16, SmallConfig1)
{
    RunDeterministicTest({8, 1, 16, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest2D_WRW_FP32, SmallConfig2)
{
    RunDeterministicTest({4, 2, 8, 8, 16, 16, 2, 2, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest2D_WRW_FP16, SmallConfig2)
{
    RunDeterministicTest({4, 2, 8, 8, 16, 16, 2, 2, 1, 1, 1, 1, 1, 1});
}

// 3D WRW Test Cases
TEST_F(DeterministicTest3D_WRW_FP32, SmallConfig1)
{
    // g  n  C  K  D  H   W   z  y  x  pad_d pad_h pad_w stride_d stride_h stride_w dilation_d
    // dilation_h dilation_w
    RunDeterministicTest({4, 1, 8, 8, 8, 14, 14, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest3D_WRW_FP16, SmallConfig1)
{
    RunDeterministicTest({4, 1, 8, 8, 8, 14, 14, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest3D_WRW_BFP16, SmallConfig1)
{
    RunDeterministicTest({4, 1, 8, 8, 8, 14, 14, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest3D_WRW_FP32, SmallConfig2)
{
    RunDeterministicTest({2, 2, 4, 4, 6, 12, 12, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

TEST_F(DeterministicTest3D_WRW_FP16, SmallConfig2)
{
    RunDeterministicTest({2, 2, 4, 4, 6, 12, 12, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

} // namespace group_conv_deterministic
